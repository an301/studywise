import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Camera, Brain, Video, VideoOff } from 'lucide-react';

// Format seconds to MM:SS
const formatTime = (seconds) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
};

const POSTURE_CONFIG = {
  neutral: { color: '#22c55e', bg: 'bg-green-500', label: 'Great posture!' },
  slouch: { color: '#f59e0b', bg: 'bg-yellow-500', label: 'Sit up straight!' },
  lean: { color: '#ef4444', bg: 'bg-red-500', label: 'Move back!' },
  away: { color: '#6b7280', bg: 'bg-gray-500', label: 'Away' },
};

export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [showVideo, setShowVideo] = useState(true);
  const [imageData, setImageData] = useState(null);
  const [posture, setPosture] = useState({ state: 'away', confidence: 0, present: false });
  const [sessionTime, setSessionTime] = useState(0);
  const [focusedTime, setFocusedTime] = useState(0);
  const [awayTime, setAwayTime] = useState(0);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  
  const wsRef = useRef(null);
  
  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    const ws = new WebSocket(`ws://localhost:8000/ws/stream`);
    
    ws.onopen = () => {
      console.log('Connected');
      setConnected(true);
      setError(null);
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.error) {
        setError(data.error);
        return;
      }
      
      if (data.type === 'frame') {
        setImageData(`data:image/jpeg;base64,${data.image}`);
        setPosture(data.posture);
        setSessionTime(data.session_seconds);
        setFocusedTime(data.focused_seconds);
        setAwayTime(data.away_seconds);
      }
    };
    
    ws.onclose = () => {
      console.log('Disconnected');
      setConnected(false);
      setImageData(null);
    };
    
    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      setError('Connection failed. Is the server running?');
      setConnected(false);
    };
    
    wsRef.current = ws;
  }, []);
  
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
    setImageData(null);
  }, []);
  
  const reset = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    }
  }, []);
  
  const toggleRunning = () => {
    if (isRunning) {
      disconnect();
    } else {
      connect();
    }
    setIsRunning(!isRunning);
  };
  
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);
  
  const config = POSTURE_CONFIG[posture.state] || POSTURE_CONFIG.away;
  const focusRate = sessionTime > 0 ? Math.round((focusedTime / sessionTime) * 100) : 0;
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-500 flex items-center justify-center">
            <Brain size={24} />
          </div>
          <h1 className="text-2xl font-bold">StudyWise</h1>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Connection status */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            connected ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'
          }`}>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-400' : 'bg-gray-400'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          
          {/* Toggle video */}
          <button
            onClick={() => setShowVideo(!showVideo)}
            className={`p-2 rounded-lg transition-colors ${
              showVideo ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-500/20 text-gray-400'
            }`}
          >
            {showVideo ? <Video size={20} /> : <VideoOff size={20} />}
          </button>
        </div>
      </header>
      
      <div className="max-w-4xl mx-auto">
        {/* Main timer */}
        <div className="text-center mb-8">
          <div className="text-gray-400 mb-2">Session Time</div>
          <div className="text-7xl font-bold font-mono tracking-tight mb-4">
            {formatTime(sessionTime)}
          </div>
          
          {/* Time breakdown */}
          <div className="flex justify-center gap-8 text-lg">
            <div>
              <span className="text-green-400 font-mono">{formatTime(focusedTime)}</span>
              <span className="text-gray-400 ml-2">focused</span>
            </div>
            <div>
              <span className="text-gray-400 font-mono">{formatTime(awayTime)}</span>
              <span className="text-gray-400 ml-2">away</span>
            </div>
            <div>
              <span className="text-blue-400 font-mono">{focusRate}%</span>
              <span className="text-gray-400 ml-2">focus rate</span>
            </div>
          </div>
        </div>
        
        {/* Video + Posture display */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Video feed */}
          {showVideo && (
            <div className="bg-black/40 rounded-2xl overflow-hidden border border-white/10">
              {imageData ? (
                <img 
                  src={imageData} 
                  alt="Camera feed" 
                  className="w-full h-auto"
                />
              ) : (
                <div className="aspect-video flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <Camera size={48} className="mx-auto mb-2 opacity-50" />
                    <p>{error || 'Click Start to begin'}</p>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Posture status */}
          <div className={`rounded-2xl p-8 border border-white/10 ${
            showVideo ? '' : 'lg:col-span-2'
          }`} style={{ backgroundColor: `${config.color}15` }}>
            <div className="text-center">
              {/* Big posture indicator */}
              <div 
                className={`inline-flex items-center gap-3 px-6 py-3 rounded-full text-2xl font-bold mb-4`}
                style={{ backgroundColor: `${config.color}30`, color: config.color }}
              >
                <span 
                  className="w-4 h-4 rounded-full animate-pulse"
                  style={{ backgroundColor: config.color }}
                />
                {posture.state.toUpperCase()}
                {posture.confidence > 0 && (
                  <span className="text-lg opacity-70">
                    {Math.round(posture.confidence * 100)}%
                  </span>
                )}
              </div>
              
              <p className="text-xl" style={{ color: config.color }}>
                {config.label}
              </p>
              
              {/* Status details */}
              <div className="mt-6 grid grid-cols-2 gap-4 text-sm">
                <div className="bg-black/20 rounded-xl p-3">
                  <div className="text-gray-400">Present</div>
                  <div className={posture.present ? 'text-green-400' : 'text-red-400'}>
                    {posture.present ? 'Yes' : 'No'}
                  </div>
                </div>
                <div className="bg-black/20 rounded-xl p-3">
                  <div className="text-gray-400">Model</div>
                  <div className={posture.model_loaded ? 'text-green-400' : 'text-yellow-400'}>
                    {posture.model_loaded ? 'Active' : 'Fallback'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Controls */}
        <div className="flex justify-center gap-4">
          <button
            onClick={toggleRunning}
            className={`flex items-center gap-2 px-8 py-4 rounded-xl font-semibold text-lg transition-all ${
              isRunning 
                ? 'bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30'
                : 'bg-green-500 text-white hover:bg-green-600 shadow-lg shadow-green-500/25'
            }`}
          >
            {isRunning ? <Pause size={24} /> : <Play size={24} />}
            {isRunning ? 'Stop' : 'Start Session'}
          </button>
          
          {isRunning && (
            <button
              onClick={reset}
              className="flex items-center gap-2 px-6 py-4 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-colors"
            >
              <RotateCcw size={20} />
              Reset
            </button>
          )}
        </div>
        
        {/* Error message */}
        {error && (
          <div className="mt-6 p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-400 text-center">
            {error}
            <div className="text-sm mt-2 text-gray-400">
              Make sure the backend server is running on port 8000
            </div>
          </div>
        )}
        
        {/* Instructions */}
        {!isRunning && !error && (
          <div className="mt-8 text-center text-gray-500 text-sm">
            <p>Click <strong>Start Session</strong> to begin tracking your posture</p>
            <p className="mt-1">Your webcam will activate and show what the model sees</p>
          </div>
        )}
      </div>
    </div>
  );
}
