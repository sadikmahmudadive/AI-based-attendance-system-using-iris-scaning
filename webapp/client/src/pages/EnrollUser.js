import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { UserPlus, Upload, Eye, CheckCircle, AlertCircle, Sparkles, User, Mail, Phone, Building, CreditCard } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

function EnrollUser() {
  const [formData, setFormData] = useState({
    iris_id: '',
    name: '',
    email: '',
    department: '',
    employee_id: '',
    phone: ''
  });
  const [preview, setPreview] = useState(null);
  const [detectedId, setDetectedId] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [enrolling, setEnrolling] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
      setDetecting(true);

      const formDataUpload = new FormData();
      formDataUpload.append('image', file);

      try {
        const response = await axios.post('/api/iris/detect', formDataUpload, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        const data = response.data.data;
        setDetectedId(data.iris_id);
        setFormData(prev => ({ ...prev, iris_id: data.iris_id }));
        toast.success(`Detected Iris ID: ${data.iris_id} (${(data.confidence * 100).toFixed(1)}%)`);
      } catch (error) {
        toast.error('Could not detect iris ID from image');
      } finally {
        setDetecting(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp'] },
    maxFiles: 1
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.iris_id || !formData.name) {
      toast.error('Iris ID and Name are required');
      return;
    }

    setEnrolling(true);

    try {
      const response = await axios.post('/api/users/enroll', formData);
      toast.success(response.data.message);
      
      // Reset form
      setFormData({
        iris_id: '',
        name: '',
        email: '',
        department: '',
        employee_id: '',
        phone: ''
      });
      setPreview(null);
      setDetectedId(null);
    } catch (error) {
      toast.error(error.response?.data?.error || 'Enrollment failed');
    } finally {
      setEnrolling(false);
    }
  };

  return (
    <div className="animate-fadeIn">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <div className="p-2 rounded-xl bg-gradient-to-r from-green-500 to-emerald-500 shadow-lg shadow-green-500/30">
            <UserPlus className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
            Enroll User
          </h1>
          <Sparkles className="w-5 h-5 text-green-500 animate-pulse" />
        </div>
        <p className="text-gray-500 text-lg">Register a new user in the iris attendance system</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Image Upload */}
        <div className="card-gradient overflow-hidden shadow-xl">
          <div className="bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 px-6 py-5">
            <h2 className="text-xl font-bold text-white flex items-center">
              <Eye className="w-6 h-6 mr-3" />
              Step 1: Upload Iris Image
            </h2>
          </div>
          <div className="p-6">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 ${
                isDragActive 
                  ? 'border-purple-500 bg-purple-50 scale-[1.02]' 
                  : 'border-gray-200 hover:border-purple-400 hover:bg-purple-50/50'
              }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <img src={preview} alt="Preview" className="max-h-48 mx-auto rounded-xl shadow-lg" />
              ) : (
                <div className="py-6">
                  <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-purple-100 to-fuchsia-100 flex items-center justify-center">
                    <Upload className="w-10 h-10 text-purple-500" />
                  </div>
                  <p className="text-xl font-semibold text-gray-700 mb-2">
                    {isDragActive ? '✨ Drop the image here' : 'Upload iris/eye image'}
                  </p>
                  <p className="text-gray-400">to auto-detect Iris ID</p>
                </div>
              )}
            </div>

            {detecting && (
              <div className="flex flex-col items-center justify-center mt-6">
                <div className="relative">
                  <div className="w-12 h-12 rounded-full border-4 border-purple-200 border-t-purple-500 animate-spin"></div>
                  <Eye className="w-5 h-5 text-purple-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
                </div>
                <p className="mt-3 text-purple-600 font-medium animate-pulse">Detecting iris ID...</p>
              </div>
            )}

            {detectedId && (
              <div className="mt-6 p-4 bg-gradient-to-r from-emerald-50 to-green-50 rounded-xl border border-emerald-200">
                <div className="flex items-center text-emerald-700">
                  <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center mr-3">
                    <CheckCircle className="w-6 h-6 text-emerald-500" />
                  </div>
                  <div>
                    <p className="text-sm text-emerald-600">Successfully Detected</p>
                    <p className="font-bold text-lg">Iris ID: {detectedId}</p>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-6 p-4 bg-gradient-to-r from-amber-50 to-yellow-50 rounded-xl border border-amber-200">
              <div className="flex items-start">
                <div className="w-10 h-10 rounded-full bg-amber-100 flex items-center justify-center mr-3 flex-shrink-0">
                  <AlertCircle className="w-5 h-5 text-amber-600" />
                </div>
                <div className="text-sm text-amber-800">
                  <p className="font-semibold mb-2">💡 Tips for best results:</p>
                  <ul className="space-y-1">
                    <li className="flex items-center"><span className="mr-2">📸</span> Take close-up photos of the eye</li>
                    <li className="flex items-center"><span className="mr-2">💡</span> Ensure good lighting</li>
                    <li className="flex items-center"><span className="mr-2">👁️</span> Keep the eye fully open</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Enrollment Form */}
        <div className="card-gradient overflow-hidden shadow-xl">
          <div className="bg-gradient-to-r from-emerald-500 via-green-500 to-teal-500 px-6 py-5">
            <h2 className="text-xl font-bold text-white flex items-center">
              <UserPlus className="w-6 h-6 mr-3" />
              Step 2: Enter Details
            </h2>
          </div>
          <div className="p-6">
            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                    <Eye className="w-4 h-4 mr-2 text-indigo-500" />
                    Iris ID <span className="text-red-500 ml-1">*</span>
                  </label>
                  <input
                    type="text"
                    name="iris_id"
                    value={formData.iris_id}
                    onChange={handleChange}
                    className="input"
                    placeholder="Auto-detected or enter manually"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                    <User className="w-4 h-4 mr-2 text-indigo-500" />
                    Full Name <span className="text-red-500 ml-1">*</span>
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="input"
                    placeholder="John Doe"
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                    <Mail className="w-4 h-4 mr-2 text-indigo-500" />
                    Email
                  </label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className="input"
                    placeholder="john@example.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2 flex items-center">
                    <Phone className="w-4 h-4 mr-2 text-indigo-500" />
                    Phone
                  </label>
                  <input
                    type="tel"
                    name="phone"
                    value={formData.phone}
                    onChange={handleChange}
                    className="input"
                    placeholder="+1234567890"
                  />
          