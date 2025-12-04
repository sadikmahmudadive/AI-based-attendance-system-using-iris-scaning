import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Eye, Home, Camera, FileText, UserPlus, Users, Menu, X, Sparkles } from 'lucide-react';

function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { path: '/', icon: Home, label: 'Dashboard' },
    { path: '/mark-attendance', icon: Camera, label: 'Attendance' },
    { path: '/reports', icon: FileText, label: 'Reports' },
    { path: '/enroll', icon: UserPlus, label: 'Enroll' },
    { path: '/users', icon: Users, label: 'Users' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar px-6 py-4">
      <div className="container mx-auto flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center space-x-3 group">
          <div className="relative">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/30 group-hover:shadow-indigo-500/50 transition-all duration-300 group-hover:scale-110">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <Sparkles className="w-4 h-4 text-amber-400 absolute -top-1 -right-1 animate-pulse" />
          </div>
          <div>
            <span className="text-xl font-bold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              Iris Attendance
            </span>
            <span className="block text-xs text-gray-400 font-medium">AI-Powered System</span>
          </div>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center space-x-2">
          {navItems.map(({ path, icon: Icon, label }) => (
            <Link
              key={path}
              to={path}
              className={`relative flex items-center space-x-2 px-5 py-3 rounded-xl transition-all duration-300 ${
                isActive(path)
                  ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg shadow-indigo-500/30'
                  : 'text-gray-600 hover:bg-gray-100/80'
              }`}
            >
              <Icon className={`w-5 h-5 ${isActive(path) ? 'animate-pulse' : ''}`} />
              <span className="font-medium">{label}</span>
              {isActive(path) && (
                <span className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-white rounded-full"></span>
              )}
            </Link>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
         