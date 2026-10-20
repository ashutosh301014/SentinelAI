from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


REQUEST_TIMEOUT_SECONDS = 8
MAX_URL_LENGTH = 2048
MAX_PAGE_BYTES = 512 * 1024


class TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title and not self.title:
            self.title = data.strip()


def validate_public_url(url: str) -> str:
    url = url.strip()

    if not url:
        raise ValueError("URL is required.")

    if len(url) > MAX_URL_LENGTH:
        raise ValueError("URL is too long.")

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are allowed.")

    if not parsed.hostname:
        raise ValueError("A valid hostname is required.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing username/password are not allowed.")

    hostname = parsed.hostname.lower()

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Localhost URLs are not allowed.")

    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port.")

    if port is not None and port not in {80, 443}:
        raise ValueError("Only ports 80 and 443 are allowed.")

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        raise ValueError("Hostname could not be resolved.")

    if not addresses:
        raise ValueError("Hostname could not be resolved.")

    for address in addresses:
        ip_text = address[4][0]

        try:
            ip = ipaddress.ip_address(ip_text)
        except ValueError:
            raise ValueError("Invalid resolved IP address.")

        if not ip.is_global:
            raise ValueError(
                "Private, local, reserved or internal network URLs are not allowed."
            )

    return url


def resolve_ips(hostname: str) -> list[str]:
    try:
        addresses = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )

        return sorted(
            {
                address[4][0]
                for address in addresses
                if address and address[4]
            }
        )

    except socket.gaierror:
        return []


class SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self):
        super().__init__()
        self.redirect_count = 0

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        absolute_url = urljoin(req.full_url, newurl)

        validate_public_url(absolute_url)

        self.redirect_count += 1

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            absolute_url,
        )


def get_page_title(content: bytes, content_type: str | None) -> str | None:
    if not content:
        return None

    if content_type and "html" not in content_type.lower():
        return None

    try:
        text = content.decode("utf-8", errors="ignore")

        parser = TitleParser()
        parser.feed(text)

        return parser.title or None

    except Exception:
        return None


def get_ssl_details(hostname: str) -> dict:
    result = {
        "enabled": True,
        "valid": False,
        "issuer": None,
        "subject": None,
        "expires_at": None,
        "days_until_expiry": None,
        "error": None,
    }

    try:
        context = ssl.create_default_context()

        with socket.create_connection(
            (hostname, 443),
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as raw_socket:

            with context.wrap_socket(
                raw_socket,
                server_hostname=hostname,
            ) as tls_socket:

                certificate = tls_socket.getpeercert()

        issuer_parts = certificate.get("issuer", ())
        subject_parts = certificate.get("subject", ())

        issuer = ", ".join(
            f"{key}={value}"
            for group in issuer_parts
            for key, value in group
        )

        subject = ", ".join(
            f"{key}={value}"
            for group in subject_parts
            for key, value in group
        )

        not_after = certificate.get("notAfter")

        if not_after:
            expiry_timestamp = ssl.cert_time_to_seconds(not_after)

            expiry = datetime.fromtimestamp(
                expiry_timestamp,
                tz=timezone.utc,
            )

            days_until_expiry = (
                expiry - datetime.now(timezone.utc)
            ).days

            result["expires_at"] = expiry.isoformat()
            result["days_until_expiry"] = days_until_expiry

        result["issuer"] = issuer or None
        result["subject"] = subject or None
        result["valid"] = True

    except Exception as exc:
        result["error"] = str(exc)

    return result


def get_security_headers(headers) -> dict:
    checks = {
        "strict_transport_security": "Strict-Transport-Security",
        "content_security_policy": "Content-Security-Policy",
        "x_content_type_options": "X-Content-Type-Options",
        "x_frame_options": "X-Frame-Options",
        "referrer_policy": "Referrer-Policy",
        "permissions_policy": "Permissions-Policy",
    }

    values = {}

    for key, header_name in checks.items():
        header_value = headers.get(header_name)

        values[key] = {
            "present": bool(header_value),
            "value": header_value,
        }

    score = sum(
        1
        for item in values.values()
        if item["present"]
    )

    return {
        "score": score,
        "total": len(checks),
        "headers": values,
    }


def detect_hosting_hint(hostname: str, headers) -> str | None:
    hostname = hostname.lower()

    if hostname.endswith(".vercel.app"):
        return "Vercel"

    if hostname.endswith(".netlify.app"):
        return "Netlify"

    if hostname.endswith(".onrender.com"):
        return "Render"

    if hostname.endswith(".railway.app"):
        return "Railway"

    server = headers.get("server")

    if server:
        return server

    return None


def _check_service_sync(url: str) -> dict:
    safe_url = validate_public_url(url)

    parsed = urlparse(safe_url)
    hostname = parsed.hostname or ""

    resolved_ips = resolve_ips(hostname)

    redirect_handler = SafeRedirectHandler()
    opener = build_opener(redirect_handler)

    request = Request(
        safe_url,
        method="GET",
        headers={
            "User-Agent": "SentinelAI-ServiceMonitor/2.0",
            "Accept": "text/html,application/json,*/*",
        },
    )

    started = time.perf_counter()

    try:
        with opener.open(
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:

            latency_ms = round(
                (time.perf_counter() - started) * 1000,
                2,
            )

            status_code = response.getcode()

            final_url = response.geturl()

            validate_public_url(final_url)

            content_type = response.headers.get(
                "Content-Type"
            )

            content_length_header = response.headers.get(
                "Content-Length"
            )

            content = response.read(MAX_PAGE_BYTES)

            if content_length_header:
                try:
                    response_size_bytes = int(
                        content_length_header
                    )
                except ValueError:
                    response_size_bytes = len(content)
            else:
                response_size_bytes = len(content)

            title = get_page_title(
                content,
                content_type,
            )

            status = (
                "healthy"
                if 200 <= status_code < 300
                else "degraded"
            )

            ssl_details = (
                get_ssl_details(urlparse(final_url).hostname or hostname)
                if urlparse(final_url).scheme == "https"
                else {
                    "enabled": False,
                    "valid": False,
                    "issuer": None,
                    "subject": None,
                    "expires_at": None,
                    "days_until_expiry": None,
                    "error": None,
                }
            )

            security_headers = get_security_headers(
                response.headers
            )

            hosting_hint = detect_hosting_hint(
                urlparse(final_url).hostname or hostname,
                response.headers,
            )

            return {
                # Existing fields — frontend compatibility
                "url": safe_url,
                "status": status,
                "http_status": status_code,
                "latency_ms": latency_ms,
                "error": None,
                "checked_at": datetime.now(
                    timezone.utc
                ).isoformat(),

                # New website intelligence
                "website": {
                    "page_title": title,
                    "domain": hostname,
                    "protocol": parsed.scheme.upper(),
                    "final_url": final_url,
                    "redirect_count": redirect_handler.redirect_count,
                    "content_type": content_type,
                    "response_size_bytes": response_size_bytes,
                    "hosting_hint": hosting_hint,
                },

                "network": {
                    "resolved_ips": resolved_ips,
                },

                "ssl": ssl_details,

                "security": security_headers,
            }

    except HTTPError as exc:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        if 400 <= exc.code < 500:
            status = "degraded"
        else:
            status = "down"

        return {
            "url": safe_url,
            "status": status,
            "http_status": exc.code,
            "latency_ms": latency_ms,
            "error": f"HTTP {exc.code}",
            "checked_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "website": {
                "page_title": None,
                "domain": hostname,
                "protocol": parsed.scheme.upper(),
                "final_url": getattr(
                    exc,
                    "url",
                    safe_url,
                ),
                "redirect_count": redirect_handler.redirect_count,
                "content_type": exc.headers.get(
                    "Content-Type"
                )
                if exc.headers
                else None,
                "response_size_bytes": None,
                "hosting_hint": None,
            },

            "network": {
                "resolved_ips": resolved_ips,
            },

            "ssl": (
                get_ssl_details(hostname)
                if parsed.scheme == "https"
                else {
                    "enabled": False,
                    "valid": False,
                    "issuer": None,
                    "subject": None,
                    "expires_at": None,
                    "days_until_expiry": None,
                    "error": None,
                }
            ),

            "security": {
                "score": 0,
                "total": 6,
                "headers": {},
            },
        }

    except (URLError, TimeoutError, OSError) as exc:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return {
            "url": safe_url,
            "status": "down",
            "http_status": None,
            "latency_ms": latency_ms,
            "error": str(exc),
            "checked_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "website": {
                "page_title": None,
                "domain": hostname,
                "protocol": parsed.scheme.upper(),
                "final_url": None,
                "redirect_count": redirect_handler.redirect_count,
                "content_type": None,
                "response_size_bytes": None,
                "hosting_hint": None,
            },

            "network": {
                "resolved_ips": resolved_ips,
            },

            "ssl": {
                "enabled": parsed.scheme == "https",
                "valid": False,
                "issuer": None,
                "subject": None,
                "expires_at": None,
                "days_until_expiry": None,
                "error": None,
            },

            "security": {
                "score": 0,
                "total": 6,
                "headers": {},
            },
        }


async def check_service(url: str) -> dict:
    return await asyncio.to_thread(
        _check_service_sync,
        url,
    )