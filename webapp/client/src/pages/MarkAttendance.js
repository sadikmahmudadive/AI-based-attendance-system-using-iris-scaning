import React, { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import { Camera, Upload, CheckCircle, XCircle, RefreshCw, Image, Eye, Sparkles, Scan, Zap } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

function MarkAttendance() {
  const [mode, setMode] = useState('upload'); // 'upload' or 'webcam'
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const webcamRef = useRef(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      processImage(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff']
    },
    maxFiles: 1
  });

  const captureFromWebcam = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setPreview(imageSrc);
      // Convert base64 to blob
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], 'webcam-capture.jpg', { type: 'image/jpeg' });
          processImage(file);
        });
    }
  }, []);

  const processImage = async (file) => {
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('/api/attendance/mark', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const data = response.data.data;
      setResult(data);

      if (data.success) {
        toast.success(`${data.entry_type} recorded for ${data.name}!`);
      } else {
        toast.error(data.message || 'Recognition failed');
      }
    } catch (error) {
      const errorMsg = error.response?.data?.error || 'Failed to process image';
      toast.error(errorMsg);
      setResult({ success: false, message: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="animate-fadeIn">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <div className="p-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 shadow-lg shadow-cyan-500/30">
            <Scan className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
            Mark Attendance
          </h1>
          <Sparkles className="w-5 h-5 text-cyan-500 animate-pulse" />
        </div>
        <p className="text-gray-500 text-lg">Upload or capture an iris image for instant recognition</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Section */}
        <div className="card-gradient overflow-hidden shadow-xl">
          <div className="bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500 px-6 py-5">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-white flex items-center">
                <Eye className="w-6 h-6 mr-2" />
                Capture Iris
              </h2>
              <div className="flex space-x-2">
                <button
                  onClick={() => setMode('upload')}
                  className={`px-4 py-2.5 rounded-xl transition-all duration-300 flex items-center space-x-2 ${
                    mode === 'upload' 
                      ? 'bg-white text-cyan-600 shadow-lg' 
                      : 'bg-white/20 text-white hover:bg-white/30'
                  }`}
                >
                  <Upload className="w-5 h-5" />
                  <span className="text-sm font-medium">Upload</span>
                </button>
                <button
                  onClick={() => setMode('webcam')}
                  className={`px-4 py-2.5 rounded-xl transition-all duration-300 flex items-center space-x-2 ${
                    mode === 'webcam' 
                      ? 'bg-white text-cyan-600 shadow-lg' 
                      : 'bg-white/20 text-white hover:bg-white/30'
                  }`}
                >
                  <Camera className="w-5 h-5" />
                  <span className="text-sm font-medium">Webcam</span>
                </button>
              </div>
            </div>
          </div>

          <div className="p-6">
            {mode === 'upload' ? (
              <div
                {...getRootProps()}
                className={`dropzone border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 ${
                  isDragActive 
                    ? 'border-cyan-500 bg-cyan-50 scale-[1.02]' 
                    : 'border-gray-200 hover:border-cyan-400 hover:bg-cyan-50/50'
                }`}
              >
                <input {...getInputProps()} />
                {preview ? (
                  <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-xl shadow-lg" />
                ) : (
                  <div className="py-8">
                    <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-cyan-100 to-blue-100 flex items-center justify-center">
                      <Image className="w-10 h-10 text-cyan-500" />
                    </div>
                    <p className="text-xl font-semibold text-gray-700 mb-2">
                      {isDragActive ? '✨ Drop the image here' : 'Drag & drop an iris image'}
                    </p>
                    <p className="text-gray-400">or click to browse files</p>
                    <div className="mt-4 flex items-center justify-center space-x-2 text-sm text-gray-400">
                      <span className="px-2 py-1 bg-gray-100 rounded">JPG</span>
                      <span className="px-2 py-1 bg-gray-100 rounded">PNG</span>
                      <span className="px-2 py-1 bg-gray-100 rounded">BMP</span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center">
                <div className="rounded-2xl overflow-hidden mb-6 shadow-lg ring-4 ring-cyan-500/20">
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    className="w-full"
                    videoConstraints={{
                      width: 640,
                      height: 480,
                      facingMode: 'user'
                    }}
                  />
                </div>
                <button
                  onClick={captureFromWebcam}
                  disabled={loading}
                  className="btn-primary px-8 py-3 text-lg flex items-center space-x-3 mx-auto"
                >
                  <Camera className="w-6 h-6" />
                  <span>Capture Photo</span>
                </button>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center mt-8">
                <div className="relative">
                  <div className="w-16 h-16 rounded-full border-4 border-cyan-200 border-t-cyan-500 animate-spin"></div>
                  <Eye className="w-6 h-6 text-cyan-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                </div>
                <p className="mt-4 text-cyan-600 font-medium animate-pulse">Analyzing iris pattern...</p>
              </div>
            )}

            {preview && !loading && (
              <button
                onClick={reset}
                className="btn-secondary flex items-center space-x-2 mx-auto mt-6"
              >
                <RefreshCw className="w-5 h-5" />
                <span>Try Another Image</span>
              </button>
            )}
          </div>
        </div>

        {/* Result Section */}
        <div className="card-gradient overflow-hidden shadow-xl">
          <div className={`px-6 py-5 ${
            result?.success 
              ? 'bg-gradient-to-r from-emerald-500 via-green-500 to-teal-500' 
              : result && !result.success
                ? 'bg-gradient-to-r from-red-500 via-rose-500 to-pink-500'
                : 'bg-gradient-to-r from-gray-400 to-gray-500'
          }`}>
            <h2 className="text-xl font-bold text-white flex items-center">
              {result?.success ? (
                <><CheckCircle className="w-6 h-6 mr-2" /> ✅ Attendance Recorded</>
              ) : result ? (
                <><XCircle className="w-6 h-6 mr-2" /> ❌ Recognition Failed</>
              ) : (
                <><Zap className="w-6 h-6 mr-2" /> Recognition Result</>
              )}
            </h2>
          </div>

          <div className="p-6">
            {result?.success ? (
              <div className="animate-fadeIn">
                <div className="text-center mb-8">
                  <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-emerald-100 to-green-100 flex items-center justify-center animate-scaleIn">
                    <CheckCircle className="w-14 h-14 text-emerald-500" />
                  </div>
                  <p className="text-emerald-600 font-semibold text-lg">Successfully Recognized!</p>
                </div>
                
                <div className="space-y-4 bg-gradient-to-br from-gray-50 to-white rounded-xl p-4">
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-500 flex items-center"><span className="mr-2">👤</span> Name</span>
                    <span className="font-bold text-gray-800 text-lg">{result.name}</span>
                  </div>
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-500 flex items-center"><span className="mr-2">🆔</span> Iris ID</span>
                    <span className="font-mono text-indigo-600 bg-indigo-50 px-3 py-1 rounded-lg">{result.iris_id}</span>
                  </div>
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-500 flex items-center"><span className="mr-2">🕐</span> Time</span>
                    <span className="font-medium">{new Date(result.timestamp).toLocaleTimeString()}</span>
                  </div>
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-500 flex items-center"><span className="mr-2">📋</span> Entry Type</span>
                    <span className={`badge ${result.entry_type === 'CHECK_IN' ? 'badge-success' : 'badge-info'}`}>
                      {result.entry_type === 'CHECK_IN' ? '📥 Check In' : '📤 Check Out'}
                    </span>
         