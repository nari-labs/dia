#!/usr/bin/env python3
"""
Simple startup script for Dia FastAPI TTS Server
"""

import argparse
import os
import sys
import subprocess

def check_environment():
    """Check if the environment is properly set up"""
    issues = []
    
    # Check if HF_TOKEN is set
    if not os.getenv("HF_TOKEN"):
        issues.append("❌ HF_TOKEN environment variable not set")
        issues.append("   Set it with: export HF_TOKEN='your_token_here'")
        issues.append("   Get token from: https://huggingface.co/settings/tokens")
    else:
        print("✅ HF_TOKEN environment variable is set")
    
    # Check if required packages are available
    try:
        import fastapi
        import uvicorn
        import torch
        print("✅ Required packages are available")
    except ImportError as e:
        issues.append(f"❌ Missing required package: {e}")
        issues.append("   Install with: pip install -e .")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
        else:
            print("ℹ️  CUDA not available, will use CPU (slower)")
    except:
        pass
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI Server")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--check-only", action="store_true", help="Only check environment, don't start server")
    
    # Debug and logging options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    parser.add_argument("--save-outputs", action="store_true", help="Save all generated audio files")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts and processing details in console")
    parser.add_argument("--retention-hours", type=int, default=24, help="File retention period in hours (default: 24)")
    
    # Performance options
    parser.add_argument("--workers", type=int, help="Number of worker threads (default: auto-detect)")
    parser.add_argument("--no-torch-compile", action="store_true", help="Disable torch.compile optimization")
    
    # Quick preset options
    parser.add_argument("--dev", action="store_true", help="Development mode (debug + save outputs + show prompts + reload)")
    parser.add_argument("--production", action="store_true", help="Production mode (optimized settings)")
    
    args = parser.parse_args()
    
    # Handle preset modes
    if args.dev:
        args.debug = True
        args.save_outputs = True
        args.show_prompts = True
        args.reload = True
        print("🔧 Development mode enabled")
    
    if args.production:
        args.debug = False
        args.save_outputs = False
        args.show_prompts = False
        args.reload = False
        print("🏭 Production mode enabled")
    
    print("🚀 Dia TTS Server Startup")
    print("=" * 40)
    
    # Show configuration
    print("📋 Server Configuration:")
    print(f"   Debug mode: {'✅' if args.debug else '❌'}")
    print(f"   Save outputs: {'✅' if args.save_outputs else '❌'}")
    print(f"   Show prompts: {'✅' if args.show_prompts else '❌'}")
    print(f"   Auto-reload: {'✅' if args.reload else '❌'}")
    print(f"   Retention: {args.retention_hours} hours")
    if args.workers:
        print(f"   Workers: {args.workers}")
    print()
    
    # Check environment
    issues = check_environment()
    
    if issues:
        print("\n⚠️  Environment Issues:")
        for issue in issues:
            print(issue)
        
        if args.check_only:
            sys.exit(1)
        
        print("\nDo you want to continue anyway? (y/N): ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    
    if args.check_only:
        print("\n✅ Environment check passed!")
        return
    
    print(f"\n🌐 Starting server on {args.host}:{args.port}")
    print("📋 SillyTavern Configuration:")
    print("   TTS Provider: OpenAI Compatible")
    print("   Model: dia")
    print("   API Key: sk-anything")
    print(f"   Endpoint URL: http://{args.host}:{args.port}/v1/audio/speech")
    print()
    print("🔗 Server endpoints:")
    print(f"   Health Check: http://{args.host}:{args.port}/health")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print(f"   Voice List: http://{args.host}:{args.port}/v1/voices")
    print(f"   Queue Stats: http://{args.host}:{args.port}/v1/queue/stats")
    if args.debug:
        print(f"   Config API: http://{args.host}:{args.port}/v1/config")
        print(f"   Generation Logs: http://{args.host}:{args.port}/v1/logs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Build command
    cmd = [
        sys.executable, "fastapi_server.py",
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    # Add flags
    if args.reload:
        cmd.append("--reload")
    if args.debug:
        cmd.append("--debug")
    if args.save_outputs:
        cmd.append("--save-outputs")
    if args.show_prompts:
        cmd.append("--show-prompts")
    if args.retention_hours != 24:
        cmd.extend(["--retention-hours", str(args.retention_hours)])
    
    # Environment variables for advanced options
    env = os.environ.copy()
    if args.workers:
        env["DIA_MAX_WORKERS"] = str(args.workers)
    if args.no_torch_compile:
        env["DIA_DISABLE_TORCH_COMPILE"] = "1"
    
    # Start the server
    try:
        if args.debug:
            print(f"🔧 Command: {' '.join(cmd)}")
            if args.workers:
                print(f"🔧 Workers: {args.workers}")
            if args.no_torch_compile:
                print("🔧 Torch compile: disabled")
            print()
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()