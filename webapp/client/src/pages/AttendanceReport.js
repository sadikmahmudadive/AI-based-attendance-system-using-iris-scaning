import React, { useState, useEffect } from 'react';
import { Calendar, Download, Users, Clock, CheckCircle, AlertTriangle, FileText, Sparkles, BarChart3 } from 'lucide-react';
import axios from 'axios';

function AttendanceReport() {
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [records, setRecords] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRecords();
  }, [date]);

  const fetchRecords = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`/api/attendance/records?date=${date}`);
      setRecords(response.data.data.records);
      setSummary(response.data.data.summary);
    } catch (error) {
      console.error('Error fetching records:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportToCSV = () => {
    if (records.length === 0) return;
    
    const headers = Object.keys(records[0]);
    const csvContent = [
      headers.join(','),
      ...records.map(record => headers.map(h => record[h]).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attendance_${date}.csv`;
    a.click();
  };

  const stats = summary ? [
    { label: 'Total Present', value: summary.total, icon: Users, gradient: 'from-indigo-500 to-blue-500', shadow: 'shadow-indigo-500/30' },
    { label: 'On Time', value: summary.on_time, icon: CheckCircle, gradient: 'from-emerald-500 to-green-500', shadow: 'shadow-emerald-500/30' },
    { label: 'Late', value: summary.late, icon: AlertTriangle, gradient: 'from-amber-500 to-orange-500', shadow: 'shadow-amber-500/30' },
    { label: 'Total Entries', value: summary.total_entries, icon: Clock, gradient: 'from-purple-500 to-pink-500', shadow: 'shadow-purple-500/30' },
  ] : [];

  return (
    <div className="animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 rounded-xl bg-gradient-to-r from-orange-500 to-red-500 shadow-lg shadow-orange-500/30">
              <BarChart3 className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              Attendance Report
            </h1>
            <Sparkles className="w-5 h-5 text-orange-500 animate-pulse" />
          </div>
          <p className="text-gray-500 text-lg">View and export attendance records</p>
        </div>
        <div className="flex items-center space-x-4 mt-4 md:mt-0">
          <div className="flex items-center bg-white rounded-xl px-5 py-3 shadow-lg border border-gray-100">
            <Calendar className="w-5 h-5 text-indigo-500 mr-3" />
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="border-none focus:outline-none font-medium text-gray-700"
            />
          </div>
          <button
            onClick={exportToCSV}
            disabled={records.length === 0}
            className="btn-primary flex items-center space-x-2 px-6 py-3"
          >
            <Download className="w-5 h-5" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat, index) => (
          <div key={index} className={`stat-card-animated card-gradient p-5 shadow-xl ${stat.shadow}`}>
            <div className="flex items-center">
              <div className={`p-4 rounded-2xl bg-gradient-to-br ${stat.gradient} mr-4 shadow-lg ${stat.shadow}`}>
                <stat.icon className="w-7 h-7 text-white" />
              </div>
              <div>
                <p className={`text-3xl font-bold bg-gradient-to-r ${stat.gradient} bg-clip-text text-transparent`}>{stat.value}</p>
                <p className="text-gray-500 text-sm font-medium">{stat.label}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Records Table */}
      <div className="card-gradient overflow-hidden shadow-xl">
        <div className="bg-gradient-to-r from-orange-500 via-red-500 to-pink-500 px-6 py-5">
          <h2 className="text-xl font-bold text-white flex items-center">
            <FileText className="w-6 h-6 mr-3" />
            Attendance Records - {new Date(date).toLocaleDateString('en-US', { 
              weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' 
            })}
          </h2>
        </div>

        <div className="p-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <div className="loading-spinner"></div>
              <p className="mt-4 text-orange-600 font-medium animate-pulse">Loading records...</p>
            </div>
          ) : records.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Time</th>
                    <th>Name</th>
                    <th>Iris ID</th>
                    <th>Department</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {records.map((record, index) => (
                    <tr key={index} className="animate-slideIn hover:bg-orange-50/50 transition-colors" style={{ animationDelay: `${index * 0.03}s` }}>
                      <td className="text-gray-400 font-medium">{index + 1}</td>
                      <td className="font-mono text-sm font-medium text-indigo-600">
                        {new Date(record.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="font-semibold text-gray-800">{record.name}</td>
                      <td>
                        <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded-lg">
                          {record.iris_id}
                        </span>
                      </td>
                      <td>
                        {record.department ? (
                          <span className="bg-gray-100 px-2 py-1 rounded-lg text-sm font-medium text-gray-600">
                            {record.department}
                          </span>
                        ) : '-'}
                      </td>
                      <td>
                        <span className={`badge ${
                          record.entry_type === 'CHECK_IN' ? 'badge-success' : 'badge-info'
                        }`}>
                          {record.entry_type === 'CHECK_IN' ? '📥 Check In' : '📤 Check Out'}
                        </span>
  