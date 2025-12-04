import React, { useState, useEffect } from 'react';
import { Users, Clock, CheckCircle, AlertTriangle, TrendingUp, Activity, Sparkles, Zap } from 'lucide-react';
import axios from 'axios';

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [recentRecords, setRecentRecords] = useState([]);

  useEffect(() => {
    fetchDashboardData();
    fetchRecentRecords();
    
    // Refresh every 30 seconds
    const interval = setInterval(() => {
      fetchDashboardData();
      fetchRecentRecords();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await axios.get('/api/dashboard');
      setData(response.data.data);
    } catch (error) {
      console.error('Error fetching dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchRecentRecords = async () => {
    try {
      const response = await axios.get('/api/attendance/records');
      setRecentRecords(response.data.data.records.slice(-10).reverse());
    } catch (error) {
      console.error('Error fetching records:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64">
        <div className="loading-spinner"></div>
        <p className="mt-4 text-indigo-600 font-medium animate-pulse">Loading dashboard...</p>
      </div>
    );
  }

  const stats = [
    { 
      label: 'Total Present', 
      value: data?.summary?.total || 0, 
      icon: Users, 
      gradient: 'from-indigo-500 to-blue-500',
      shadowColor: 'shadow-indigo-500/30'
    },
    { 
      label: 'On Time', 
      value: data?.summary?.on_time || 0, 
      icon: CheckCircle, 
      gradient: 'from-emerald-500 to-teal-500',
      shadowColor: 'shadow-emerald-500/30'
    },
    { 
      label: 'Late', 
      value: data?.summary?.late || 0, 
      icon: AlertTriangle, 
      gradient: 'from-amber-500 to-orange-500',
      shadowColor: 'shadow-amber-500/30'
    },
    { 
      label: 'Enrolled Users', 
      value: data?.enrolled_count || 0, 
      icon: TrendingUp, 
      gradient: 'from-purple-500 to-pink-500',
      shadowColor: 'shadow-purple-500/30'
    },
  ];

  return (
    <div className="animate-fadeIn">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center space-x-3 mb-2">
          <div className="p-2 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-500 shadow-lg shadow-indigo-500/30">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
            Dashboard
          </h1>
          <Sparkles className="w-5 h-5 text-amber-500 animate-pulse" />
        </div>
        <p className="text-gray-500 text-lg">
          {new Date().toLocaleDateString('en-US', { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
          })}
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <div 
            key={index} 
            className={`stat-card-animated card-gradient p-6 animate-fadeIn ${stat.shadowColor} shadow-xl`} 
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-500 text-sm font-medium uppercase tracking-wider">{stat.label}</p>
                <p className={`text-5xl font-bold mt-2 bg-gradient-to-r ${stat.gradient} bg-clip-text text-transparent`}>
                  {stat.value}
                </p>
              </div>
              <div className={`p-4 rounded-2xl bg-gradient-to-br ${stat.gradient} shadow-lg ${stat.shadowColor}`}>
                <stat.icon className="w-8 h-8 text-white" />
              </div>
            </div>
            {/* Progress indicator */}
            <div className="mt-4 h-1.5 bg-gray-100 rounded-full overflow-hidden">
              <div 
                className={`h-full bg-gradient-to-r ${stat.gradient} rounded-full transition-all duration-1000`}
                style={{ width: `${Math.min((stat.value / 20) * 100, 100)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Attendance */}
      <div className="card-gradient overflow-hidden shadow-xl">
        <div className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 px-6 py-5">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white flex items-center">
              <Clock className="w-6 h-6 mr-3" />
              Recent Attendance
            </h2>
            <div className="flex items-center space-x-2 text-white/80 text-sm">
              <Zap className="w-4 h-4" />
              <span>Live Updates</span>
            </div>
          </div>
        </div>
        <div className="p-6">
          {recentRecords.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Name</th>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRecords.map((record, index) => (
                    <tr key={index} className="animate-slideIn hover:bg-indigo-50/50 transition-colors" style={{ animationDelay: `${index * 0.05}s` }}>
                      <td className="font-mono text-sm font-medium text-indigo-600">
                        {new Date(record.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="font-semibold text-gray-800">{record.name}</td>
                      <td className="text-gray-500 font-mono text-sm">{record.iris_id}</td>
                      <td>
                        <span className={`badge ${record.entry_type === 'CHECK_IN' ? 'badge-success' : 'badge-info'}`}>
                          {record.entry_type === 'CH