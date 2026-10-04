"use client";

import React, { useState, useRef } from "react";

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      handleFileSelection(file);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (file: File) => {
    setError(null);
    setTranscript(null);
    
    // Check if it's a valid video file
    if (!file.name.match(/\.(mp4|mpg|avi)$/i)) {
      setError("Please select a valid video file (.mp4, .mpg, .avi)");
      return;
    }

    setVideoFile(file);
    const url = URL.createObjectURL(file);
    setVideoPreviewUrl(url);
  };

  const clearSelection = () => {
    setVideoFile(null);
    setVideoPreviewUrl(null);
    setTranscript(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const processVideo = async () => {
    if (!videoFile) return;

    setIsProcessing(true);
    setError(null);
    setTranscript(null);

    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to transcribe video");
      }

      setTranscript(data.transcript);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>SilentSpeak</h1>
        <p>Visual speech recognition engine. Upload a silent video to see the AI read their lips.</p>
      </header>

      <main className="main-content">
        <section className="panel">
          {!videoFile ? (
            <div 
              className={`upload-area ${isDragging ? 'drag-active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileInput} 
                accept=".mp4,.mpg,.avi"
              />
              <div className="upload-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
              </div>
              <div className="upload-text">
                <h3>Select a video</h3>
                <p>Drag and drop or click to browse</p>
                <p style={{fontSize: '0.8rem', marginTop: '0.5rem', opacity: 0.5}}>Supports MP4, MPG, AVI</p>
              </div>
            </div>
          ) : (
            <div className="preview-container">
              <div className="video-preview">
                <video src={videoPreviewUrl!} controls loop playsInline />
              </div>
              
              <button 
                className="btn-primary" 
                onClick={processVideo} 
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <>
                    <div className="spinner"></div>
                    Processing Video...
                  </>
                ) : (
                  <>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    Read Lips
                  </>
                )}
              </button>
              
              <div style={{textAlign: 'center'}}>
                <button 
                  className="btn-secondary" 
                  onClick={clearSelection}
                  disabled={isProcessing}
                >
                  Choose Different Video
                </button>
              </div>

              {isProcessing && (
                <div className="status-text">
                  Extracting visual frames and running deep learning inference...
                </div>
              )}

              {error && (
                <div className="result-box" style={{borderColor: 'rgba(239, 68, 68, 0.5)', background: 'rgba(239, 68, 68, 0.1)'}}>
                  <h4 style={{color: '#ef4444'}}>Error</h4>
                  <p style={{color: '#fca5a5'}}>{error}</p>
                </div>
              )}

              {transcript && (
                <div className="result-box">
                  <h4>Predicted Transcript</h4>
                  <div className="transcript-text">{transcript}</div>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
