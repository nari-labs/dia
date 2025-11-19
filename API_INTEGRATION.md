# Dia TTS API Integration Guide

## What's New

- **Role-Based Input**: Automatically map chat roles (user/assistant/system) to speaker tags
- **Improved Voice Cloning**: Now follows Dia's reference implementation exactly
- **Persistent Audio Prompts**: Audio files are stored on disk and persist across restarts
- **Audio Prompt Transcripts**: Include transcripts for significantly better voice cloning

### Migration Notes

If you were using a previous version:
1. Audio prompts are now stored as files in `audio_prompts/` directory
2. The `/audio_prompts/re-encode` endpoint has been removed (no longer needed)
3. Voice mappings now support `audio_prompt_transcript` field for better results
4. The `role` parameter is now available for automatic speaker tag mapping

## Overview

Dia TTS FastAPI server provides a comprehensive text-to-speech API with advanced features:
- **Voice cloning** with audio prompts (following Dia's reference implementation)
- **Role-based input** for chat applications (user/assistant/system)
- **Configurable model parameters** for quality control
- **Debug logging** and file management
- **Custom voice creation** and management
- **SillyTavern integration** support
- **Persistent audio prompt storage** across server restarts

## Quick Start

### 1. Start the Server
```bash
# Recommended: Use startup script (includes environment check)
python start_server.py

# Development mode (debug + save outputs + prompts + reload)
python start_server.py --dev

# Production mode (optimized settings)
python start_server.py --production

# Custom configuration
python start_server.py --debug --save-outputs --workers 8 --retention-hours 48

# Direct server launch (bypass startup script)
python fastapi_server.py --debug --save-outputs --port 7860
```

### 2. Basic TTS Request
```bash
# Synchronous request (immediate response)
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test message.",
    "voice_id": "alloy",
    "role": "user",
    "response_format": "wav",
    "temperature": 1.0,
    "cfg_scale": 2.5
  }' \
  --output speech.wav

# With role-based input (new!)
curl -X POST "http://localhost:7860/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I can help you with that request.",
    "voice_id": "nova",
    "role": "assistant"
  }' \
  --output assistant_speech.wav

# Asynchronous request (job-based)
curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test message.",
    "voice_id": "alloy",
    "role": "user"
  }'
```

## Core API Endpoints

### Text-to-Speech
- **POST** `/generate` - Generate speech from text (with model parameters)
- **GET** `/models` - List available models
- **GET** `/voices` - List available voices

### Voice Management
- **GET** `/voice_mappings` - List voice configurations
- **POST** `/voice_mappings` - Create custom voice
- **PUT** `/voice_mappings/{voice_id}` - Update voice
- **DELETE** `/voice_mappings/{voice_id}` - Delete custom voice

### Audio Prompts
- **POST** `/audio_prompts/upload` - Upload voice sample (saved as WAV file)
- **GET** `/audio_prompts` - List uploaded prompts with file info
- **DELETE** `/audio_prompts/{prompt_id}` - Delete prompt and file

### Debug & Configuration
- **GET** `/config` - Get server configuration
- **PUT** `/config` - Update server configuration
- **GET** `/logs` - List generation logs
- **GET** `/logs/{id}/download` - Download saved audio
- **POST** `/cleanup` - Manual file cleanup

### Job Queue Management
- **GET** `/jobs` - List jobs with status filtering
- **GET** `/jobs/{id}` - Get job status and details
- **GET** `/jobs/{id}/result` - Download completed job result
- **DELETE** `/jobs/{id}` - Cancel pending job
- **GET** `/queue/stats` - Get queue statistics
- **DELETE** `/jobs` - Clear completed jobs

## SillyTavern Integration

### Configuration
1. Navigate to **Settings → Text-to-Speech**
2. Set **TTS Provider**: `Custom`
3. Set **Endpoint URL**: `http://localhost:7860/generate`
4. Choose **Voice**: Built-in (`alloy`, `echo`, `fable`, `nova`, `onyx`, `shimmer`) or custom voices

### Advanced Features
- **Custom Voices**: Upload audio samples and create character-specific voices
- **Model Parameters**: SillyTavern may pass through additional parameters
- **Debug Mode**: Enable server debug logging to troubleshoot issues
- **Role Support**: API automatically maps chat roles to appropriate speakers

### Role-Based Generation Example
When SillyTavern sends requests with roles, they are automatically mapped:
```json
// User message (maps to [S1])
{
  "text": "Tell me a story about dragons.",
  "voice_id": "alloy",
  "role": "user"
}

// Character/Assistant response (maps to [S2])
{
  "text": "Once upon a time, in a land far away...",
  "voice_id": "nova",
  "role": "assistant"
}
```

### Custom Voice Setup
```bash
# 1. Upload voice sample
curl -X POST "http://localhost:7860/audio_prompts/upload" \
  -F "prompt_id=character_voice" \
  -F "audio_file=@sample.wav"

# 2. Create voice mapping with transcript
curl -X POST "http://localhost:7860/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "my_character",
    "style": "expressive",
    "primary_speaker": "S1", 
    "audio_prompt": "character_voice",
    "audio_prompt_transcript": "[S1] Sample text from the character."
  }'

# 3. Use in SillyTavern by setting Voice: "my_character"
```

## Request/Response Formats

### TTS Request
```json
{
  "text": "Your text here",
  "voice_id": "alloy",
  "response_format": "wav",
  "speed": 1.0,
  "role": "user",  // Optional: "user", "assistant", or "system"
  "temperature": 1.2,
  "cfg_scale": 3.0,
  "top_p": 0.95,
  "max_tokens": 2000,
  "use_torch_compile": true
}
```

### Voice Mapping
```json
{
  "voice_id": "custom_voice",
  "style": "neutral",
  "primary_speaker": "S1",
  "audio_prompt": "prompt_id",
  "audio_prompt_transcript": "[S1] This is what I said in the audio sample."
}
```

## Text Format Guidelines

### Role-Based Input (NEW)
The API now supports role-based text formatting for easier integration with chat applications:
- **`role: "user"`** - Maps to `[S1]` speaker tag
- **`role: "assistant"`** - Maps to `[S2]` speaker tag  
- **`role: "system"`** - Maps to `[S2]` speaker tag

Example with roles:
```json
// User speaking
{
  "text": "What's the weather like today?",
  "voice_id": "alloy",
  "role": "user"
}

// Assistant responding
{
  "text": "The weather is sunny with a high of 75 degrees.",
  "voice_id": "nova",
  "role": "assistant"
}
```

### Speaker Tags (Manual)
- Always start with `[S1]` or `[S2]`
- Alternate speakers: `[S1] Hello! [S2] Hi there!`
- Single speaker: `[S1] This is a monologue.`

### Nonverbal Sounds
- `(laughs)`, `(coughs)`, `(sighs)`, `(gasps)`
- `(clears throat)`, `(whispers)`, `(shouts)`

### Best Practices
- Keep input under 4096 characters
- Use clear, natural speech patterns
- For voice cloning: provide 3-10 seconds of clean audio

## Error Handling

### Common Status Codes
- `400` - Invalid request format
- `404` - Voice/prompt not found
- `500` - Generation or server error

### Example Error Response
```json
{
  "detail": "Voice 'unknown_voice' not found"
}
```

## Asynchronous Processing

### Job-Based Workflow
```bash
# 1. Submit job
response=$(curl -X POST "http://localhost:7860/generate?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[S1] Hello world!",
    "voice_id": "alloy"
  }')

job_id=$(echo $response | jq -r '.job_id')

# 2. Check job status
curl "http://localhost:7860/jobs/$job_id"

# 3. Download result when completed
curl "http://localhost:7860/jobs/$job_id/result" -o output.wav

# 4. Get queue statistics
curl "http://localhost:7860/queue/stats"
```

### Job Management
```bash
# List all jobs
curl "http://localhost:7860/jobs"

# List only pending jobs
curl "http://localhost:7860/jobs?status=pending"

# Cancel a pending job
curl -X DELETE "http://localhost:7860/jobs/$job_id"

# Clear completed jobs
curl -X DELETE "http://localhost:7860/jobs"
```

## Advanced Configuration

### Server Configuration
```bash
# Enable debug features
curl -X PUT "http://localhost:7860/config" \
  -H "Content-Type: application/json" \
  -d '{
    "debug_mode": true,
    "save_outputs": true,
    "show_prompts": true,
    "output_retention_hours": 48
  }'

# Get current configuration
curl "http://localhost:7860/config"
```

### Generation Monitoring
```bash
# View recent generations
curl "http://localhost:7860/logs?limit=10"

# Download specific generation
curl "http://localhost:7860/logs/{log_id}/download" -o output.wav

# Clear all logs
curl -X DELETE "http://localhost:7860/logs"
```

## Voice Cloning Workflow (Enhanced)

### 1. Prepare Audio Sample
- **Formats**: WAV, MP3, M4A, FLAC, OGG, AAC (auto-converted to WAV)
- **Duration**: 0.5-60 seconds (3-10 seconds recommended)
- **Quality**: Clear speech, minimal background noise
- **Content**: Speaker saying target text or similar phrases
- **Storage**: Audio files are saved in `audio_prompts/` directory as WAV files
- **Persistence**: Audio prompts persist across server restarts (loaded on startup)

### 2. Upload and Configure
```python
import requests

# Upload audio prompt (saved as WAV file)
with open("voice_sample.wav", "rb") as f:
    response = requests.post(
        "http://localhost:7860/audio_prompts/upload",
        data={"prompt_id": "my_voice"},
        files={"audio_file": f}
    )

# Create voice mapping WITH transcript for better results
# IMPORTANT: The transcript should match what's in the audio file
requests.post(
    "http://localhost:7860/voice_mappings",
    json={
        "voice_id": "cloned_voice",
        "style": "natural",
        "primary_speaker": "S1",
        "audio_prompt": "my_voice",
        "audio_prompt_transcript": "[S1] This is what I said in the audio sample."
    }
)

# The model will:
# 1. Load the audio file directly
# 2. Prepend the transcript to your text
# 3. Generate audio only for the new text portion
```

### 3. Generate Speech
```python
response = requests.post(
    "http://localhost:7860/generate",
    json={
        "voice_id": "alloy",
        "text": "[S1] Hello, this is my cloned voice!",
        "voice_id": "cloned_voice",
        "temperature": 1.1,
        "cfg_scale": 2.8,
        "top_p": 0.9
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### How Voice Cloning Works

Based on the Dia model's reference implementation:

1. **Audio Prompt**: The model loads your audio file directly
2. **Transcript Concatenation**: Your audio transcript is prepended to the target text
3. **Generation**: The model generates audio for the ENTIRE concatenated text
4. **Output**: Only the audio for the new text (after transcript) is returned

Example flow:
```
Audio prompt transcript: "[S1] Hi, this is my voice sample."
Target text: "[S1] Hello world!"
Model input: "[S1] Hi, this is my voice sample. [S1] Hello world!"
Generated output: Audio for "Hello world!" in the cloned voice
```

**Best Practices**:
- Ensure transcript exactly matches the audio content
- Use the same speaker tag ([S1] or [S2]) consistently
- Keep audio samples clear and noise-free
- 3-10 seconds of speech works best

### Complete Voice Cloning Example with Roles

```python
import requests

# 1. Upload character voice sample
with open("character_voice.wav", "rb") as f:
    requests.post(
        "http://localhost:7860/audio_prompts/upload",
        data={"prompt_id": "my_character"},
        files={"audio_file": f}
    )

# 2. Create voice mapping with transcript
requests.post(
    "http://localhost:7860/voice_mappings",
    json={
        "voice_id": "ai_assistant",
        "style": "friendly",
        "primary_speaker": "S2",  # Assistant uses S2
        "audio_prompt": "my_character",
        "audio_prompt_transcript": "[S2] Hello, I am your AI assistant."
    }
)

# 3. Generate responses with role-based input
# User question (uses S1 automatically)
user_response = requests.post(
    "http://localhost:7860/generate",
    json={
        "text": "Can you help me with Python?",
        "voice_id": "alloy",
        "role": "user"  # Maps to S1
    }
)

# AI assistant response (uses cloned voice)
assistant_response = requests.post(
    "http://localhost:7860/generate", 
    json={
        "text": "Of course! I'd be happy to help you with Python programming.",
        "voice_id": "ai_assistant",
        "role": "assistant"  # Maps to S2
    }
)
```

## Integration Examples

### Python Client
```python
import requests

class DiaTTSClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url
    
    def generate_speech(self, text, voice="alloy", **kwargs):
        payload = {
            "voice_id": "alloy",
            "text": text,
            "voice_id": voice
        }
        # Add any additional parameters
        payload.update(kwargs)
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload
        )
        return response.content
    
    def upload_voice(self, prompt_id, audio_file_path):
        with open(audio_file_path, "rb") as f:
            return requests.post(
                f"{self.base_url}/audio_prompts/upload",
                data={"prompt_id": prompt_id},
                files={"audio_file": f}
            ).json()

# Usage
client = DiaTTSClient()
audio = client.generate_speech(
    "[S1] Hello world!", 
    "alloy",
    temperature=1.1,
    cfg_scale=2.5,
    top_p=0.9
)
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');

class DiaTTSClient {
    constructor(baseUrl = 'http://localhost:7860') {
        this.baseUrl = baseUrl;
    }
    
    async generateSpeech(text, voice = 'alloy', options = {}) {
        const payload = {
            model: 'dia',
            input: text,
            voice: voice,
            ...options
        };
        
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        return response.arrayBuffer();
    }
    
    async generateSpeechAsync(text, voice = 'alloy', options = {}) {
        const payload = {
            model: 'dia',
            input: text,
            voice: voice,
            ...options
        };
        
        // Submit job
        const response = await fetch(`${this.baseUrl}/generate?async_mode=true`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const { job_id } = await response.json();
        
        // Poll for completion
        while (true) {
            const statusResponse = await fetch(`${this.baseUrl}/jobs/${job_id}`);
            const job = await statusResponse.json();
            
            if (job.status === 'completed') {
                const resultResponse = await fetch(`${this.baseUrl}/jobs/${job_id}/result`);
                return resultResponse.arrayBuffer();
            } else if (job.status === 'failed') {
                throw new Error(`Job failed: ${job.error_message}`);
            }
            
            // Wait 1 second before polling again
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    
    async getQueueStats() {
        const response = await fetch(`${this.baseUrl}/queue/stats`);
        return response.json();
    }
}

// Usage examples
const client = new DiaTTSClient();

// Synchronous generation
const syncAudio = await client.generateSpeech('[S1] Hello world!', 'alloy', {
    temperature: 1.0,
    cfg_scale: 2.5
});

// Asynchronous generation
const asyncAudio = await client.generateSpeechAsync('[S1] Long text...', 'nova');

// Monitor queue
const stats = await client.getQueueStats();
console.log(`Pending jobs: ${stats.pending_jobs}, Active workers: ${stats.active_workers}`);
```

## Model Parameters

### Available Parameters
- **temperature** (0.1-2.0): Controls randomness in generation. Higher = more creative, lower = more consistent
- **cfg_scale** (1.0-10.0): Classifier-free guidance scale. Higher = stronger conditioning on input text
- **top_p** (0.0-1.0): Nucleus sampling threshold. Lower = more focused sampling
- **max_tokens** (100-10000): Maximum number of tokens to generate
- **use_torch_compile** (boolean): Enable PyTorch compilation for faster inference

### Default Values
- temperature: 1.2
- cfg_scale: 3.0
- top_p: 0.95
- max_tokens: auto (based on text length)
- use_torch_compile: auto-detected

### Parameter Effects
- **Lower temperature + higher cfg_scale**: More consistent, text-faithful speech
- **Higher temperature + lower cfg_scale**: More expressive, natural-sounding speech
- **Lower top_p**: More focused vocabulary, clearer pronunciation
- **Higher top_p**: More varied expressions and natural speech patterns

### Recommended Settings
```json
// High quality, consistent output
{
  "temperature": 0.8,
  "cfg_scale": 4.0,
  "top_p": 0.8
}

// Natural, expressive speech
{
  "temperature": 1.4,
  "cfg_scale": 2.5,
  "top_p": 0.95
}

// Fast generation
{
  "temperature": 1.0,
  "cfg_scale": 2.0,
  "max_tokens": 1000,
  "use_torch_compile": true
}
```

## File Storage Structure

```
dia/
├── audio_prompts/           # Uploaded voice samples (persistent)
│   ├── my_voice.wav
│   ├── character_voice.wav
│   └── ...
├── audio_outputs/           # Generated audio (when save_outputs=true)
│   ├── 20240115_123456_abc123_alloy.wav
│   └── ...
└── fastapi_server.py
```

## Performance Considerations

- **Model Loading**: ~30 seconds on first request
- **Generation Speed**: ~2-5x real-time on GPU
- **Memory Usage**: ~10GB VRAM (GPU) or ~16GB RAM (CPU)
- **Concurrent Requests**: Up to 4 workers by default (configurable)
- **Parameter Impact**: Higher temperature/top_p may increase generation time slightly
- **Queue Benefits**: Non-blocking request handling, better resource utilization
- **Worker Scaling**: Limited by GPU memory (single model instance shared)

## Troubleshooting

### Common Issues
1. **Model not loading**: Check `HF_TOKEN` environment variable
2. **Audio quality**: Ensure input text has proper speaker tags (`[S1]`, `[S2]`)
3. **Slow generation**: Use GPU with CUDA for faster inference
4. **Voice not found**: Check `/voice_mappings` for available voices
5. **Torch compile errors**: Server auto-retries without compilation
6. **File upload failures**: Check audio format and file size limits
7. **Queue full**: Use async mode or check `/queue/stats` for worker availability
8. **Job not found**: Jobs expire after 1 hour, check job list with `/jobs`
9. **Audio prompt not influencing output**: 
   - Ensure audio file exists in audio_prompts directory
   - Include accurate audio prompt transcript (must match audio content)
   - The transcript is prepended to your text for conditioning
   - Model generates audio only for text after the transcript

### Debug Commands
```bash
# Health check
curl "http://localhost:7860/health"

# List available models
curl "http://localhost:7860/models"

# List available voices
curl "http://localhost:7860/voices"

# Check voice mappings
curl "http://localhost:7860/voice_mappings"

# View server configuration
curl "http://localhost:7860/config"

# Check queue statistics
curl "http://localhost:7860/queue/stats"

# List current jobs
curl "http://localhost:7860/jobs"

# Enable debug mode
curl -X PUT "http://localhost:7860/config" \
  -d '{"debug_mode": true}' \
  -H "Content-Type: application/json"
```

### Log Analysis
- Monitor server console for detailed error information
- Check generation logs via `/logs` endpoint
- Enable `show_prompts` to see text processing
- Use `save_outputs` to verify audio generation

### Performance Tuning
- Use `use_torch_compile: false` on Windows or unsupported systems
- Lower `max_tokens` for faster generation
- Reduce `temperature` and `cfg_scale` for simpler processing
- Enable debug logging to identify bottlenecks
- Use async mode (`?async_mode=true`) for concurrent requests
- Monitor queue stats to optimize worker utilization
- Consider increasing worker count for high-throughput scenarios