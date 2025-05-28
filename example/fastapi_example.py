"""
Example usage of the Dia FastAPI server
"""

import requests
import io
import soundfile as sf

# Server configuration
SERVER_URL = "http://localhost:8000"

def test_tts_api():
    """Test the TTS API endpoint"""
    
    # Test text
    text = "Hello, this is a test of the Dia text-to-speech system. How does this sound?"
    
    # Make request to TTS endpoint
    response = requests.post(
        f"{SERVER_URL}/v1/audio/speech",
        json={
            "model": "dia",
            "input": text,
            "voice": "alloy",
            "response_format": "wav",
            "speed": 1.0
        }
    )
    
    if response.status_code == 200:
        # Save the audio file
        with open("fastapi_test_output.wav", "wb") as f:
            f.write(response.content)
        print("✅ TTS generation successful! Audio saved to 'fastapi_test_output.wav'")
        
        # Load and print audio info
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        duration = len(audio_data) / sample_rate
        print(f"📊 Audio info: {duration:.2f}s duration, {sample_rate}Hz sample rate")
    else:
        print(f"❌ TTS generation failed: {response.status_code}")
        print(f"Error: {response.text}")

def test_voices_api():
    """Test the voices listing endpoint"""
    
    response = requests.get(f"{SERVER_URL}/v1/voices")
    
    if response.status_code == 200:
        voices = response.json()
        print("✅ Available voices:")
        for voice in voices:
            print(f"  - {voice['name']} (ID: {voice['voice_id']})")
    else:
        print(f"❌ Failed to get voices: {response.status_code}")

def test_health():
    """Test server health"""
    
    response = requests.get(f"{SERVER_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print("✅ Server health:")
        print(f"  Status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

def test_alternative_endpoint():
    """Test the alternative TTS endpoint (SillyTavern-Extras style)"""
    
    response = requests.post(
        f"{SERVER_URL}/api/tts/generate",
        json={
            "text": "This is a test using the alternative API endpoint.",
            "speaker": "nova"
        }
    )
    
    if response.status_code == 200:
        with open("fastapi_alt_test_output.wav", "wb") as f:
            f.write(response.content)
        print("✅ Alternative API test successful! Audio saved to 'fastapi_alt_test_output.wav'")
    else:
        print(f"❌ Alternative API test failed: {response.status_code}")

if __name__ == "__main__":
    print("🧪 Testing Dia FastAPI Server")
    print("=" * 40)
    
    # Test server health first
    test_health()
    print()
    
    # Test voices endpoint
    test_voices_api()
    print()
    
    # Test main TTS endpoint
    test_tts_api()
    print()
    
    # Test alternative endpoint
    test_alternative_endpoint()
    print()
    
    print("🎉 All tests completed!")
    print()
    print("💡 SillyTavern Configuration:")
    print("  TTS Provider: OpenAI Compatible")
    print("  Model: dia")
    print("  API Key: not-needed")
    print(f"  Endpoint URL: {SERVER_URL}/v1/audio/speech")