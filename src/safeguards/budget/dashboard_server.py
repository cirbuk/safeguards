"""HTTP server for budget monitoring dashboard."""

from pathlib import Path

from aiohttp import web
from aiohttp.web import Application, Request, Response


class DashboardServer:
    """Serves the budget monitoring dashboard."""

    def __init__(self, host: str = "localhost", port: int = 8080):
        """Initialize dashboard server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.app = Application()
        self._setup_routes()

    def _setup_routes(self):
        """Set up HTTP routes."""
        self.app.router.add_get("/", self.serve_dashboard)

        # Add static files route
        static_dir = Path(__file__).parent / "templates"
        self.app.router.add_static("/static/", static_dir)

    async def serve_dashboard(self, request: Request) -> Response:
        """Serve the dashboard HTML.

        Args:
            request: HTTP request

        Returns:
            HTTP response with dashboard HTML
        """
        template_path = Path(__file__).parent / "templates" / "dashboard.html"
        if not template_path.exists():
            return Response(text="Dashboard template not found", status=404)

        with open(template_path) as f:
            html = f.read()
        return Response(text=html, content_type="text/html")

    async def start(self):
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"Dashboard server running at http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP server."""
        await self.app.shutdown()
        await self.app.cleanup()
