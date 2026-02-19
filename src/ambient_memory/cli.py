"""
Command line interface for Ambient Memory server.
"""

import typer
import uvicorn
from pathlib import Path
from .server import create_app

app = typer.Typer(name="ambient-memory", help="Memory for AI agents that actually works")


@app.command("serve")
def serve(
    port: int = typer.Option(9876, "--port", "-p", help="Port to run server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    chroma_path: str = typer.Option("./data", "--chroma-path", help="Path to ChromaDB data directory"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the Ambient Memory server."""
    
    # Ensure data directory exists
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"🧠 Starting Ambient Memory server on http://{host}:{port}")
    typer.echo(f"📁 ChromaDB data: {chroma_path}")
    
    # Create the FastAPI app with specified chroma path
    fastapi_app = create_app(chroma_path=chroma_path)
    
    # Start the server
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        reload=reload,
        access_log=True,
    )


@app.command("version")
def version():
    """Show version information."""
    from . import __version__, __author__
    typer.echo(f"Ambient Memory v{__version__}")
    typer.echo(f"Built by {__author__}")


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()