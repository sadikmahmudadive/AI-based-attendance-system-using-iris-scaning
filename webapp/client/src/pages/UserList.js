import React, { useState, useEffect } from 'react';
import { Users, Trash2, Search, UserCheck, UserX, Sparkles, Shield, Eye } from 'lucide-react';
import axios from 'axios';
import toast from 'react-hot-toast';

function UserList() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const response = await axios.get('/api/users');
      setUsers(response.data.data);
    } catch (error) {
      toast.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (irisId, name) => {
    try {
      await axios.delete(`/api/users/${irisId}`);
      toast.success(`Deleted ${name}`);
      setUsers(users.filter(u => u.iris_id !== irisId));
      setDeleteConfirm(null);
    } catch (error) {
      toast.error('Failed to delete user');
    }
  };

  const filteredUsers = users.filter(user =>
    user.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.iris_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.department?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-500 shadow-lg shadow-blue-500/30">
              <Users className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              Enrolled Users
            </h1>
            <Sparkles className="w-5 h-5 text-blue-500 animate-pulse" />
          </div>
          <p className="text-gray-500 text-lg">Manage registered users in the system</p>
        </div>
        <div className="mt-4 md:mt-0">
          <div className="relative">
            <Search className="w-5 h-5 absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search users..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-12 pr-4 py-3 rounded-xl border-2 border-gray-200 focus:outline-none focus:border-indigo-500 w-72 bg-white/80 backdrop-blur-sm shadow-lg transition-all duration-300"
            />
          </div>
        </div>
      </div>

      {/* User Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
        <div className="stat-card-animated card-gradient p-5 flex items-center shadow-xl shadow-indigo-500/10">
          <div className="p-4 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-2xl mr-4 shadow-lg shadow-indigo-500/30">
            <Users className="w-7 h-7 text-white" />
          </div>
          <div>
            <p className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">{users.length}</p>
            <p className="text-gray-500 text-sm font-medium">Total Users</p>
          </div>
        </div>
        <div className="stat-card-animated card-gradient p-5 flex items-center shadow-xl shadow-green-500/10">
          <div className="p-4 bg-gradient-to-br from-emerald-500 to-green-500 rounded-2xl mr-4 shadow-lg shadow-emerald-500/30">
            <UserCheck className="w-7 h-7 text-white" />
          </div>
          <div>
            <p className="text-3xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">{users.filter(u => u.is_active).length}</p>
            <p className="text-gray-500 text-sm font-medium">Active</p>
          </div>
        </div>
        <div className="stat-card-animated card-gradient p-5 flex items-center shadow-xl shadow-cyan-500/10">
          <div className="p-4 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-2xl mr-4 shadow-lg shadow-cyan-500/30">
            <Eye className="w-7 h-7 text-white" />
          </div>
          <div>
            <p className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">{users.length}</p>
            <p className="text-gray-500 text-sm font-medium">Iris Scans</p>
          </div>
        </div>
        <div className="stat-card-animated card-gradient p-5 flex items-center shadow-xl shadow-amber-500/10">
          <div className="p-4 bg-gradient-to-br from-amber-500 to-orange-500 rounded-2xl mr-4 shadow-lg shadow-amber-500/30">
            <Shield className="w-7 h-7 text-white" />
          </div>
          <div>
            <p className="text-3xl font-bold bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent">100%</p>
            <p className="text-gray-500 text-sm font-medium">Secured</p>
          </div>
        </div>
      </div>

      {/* Users Table */}
      <div className="card-gradient overflow-hidden shadow-xl">
        <div className="bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 px-6 py-5">
          <h2 className="text-xl font-bold text-white flex items-center">
            <Users className="w-6 h-6 mr-3" />
            All Enrolled Users
            <span className="ml-3 px-3 py-1 bg-white/20 rounded-full text-sm font-medium">{filteredUsers.length} users</span>
          </h2>
        </div>

        <div className="p-6">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16">
              <div className="loading-spinner"></div>
              <p className="mt-4 text-indigo-600 font-medium animate-pulse">Loading users...</p>
            </div>
          ) : filteredUsers.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Iris ID</th>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Department</th>
                    <th>Employee ID</th>
                    <th>Enrolled</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredUsers.map((user, index) => (
                    <tr key={user.iris_id} className="animate-slideIn hover:bg-indigo-50/50 transition-colors" style={{ animationDelay: `${index * 0.03}s` }}>
                      <td className="text-gray-400 font-medium">{index + 1}</td>
                      <td>
                        <span className="font-mono font-semibold text-indigo-600 bg-indigo-50 px-2 py-1 rounded-lg text-sm">
                          {user.iris_id}
                        </span>
                      </td>
                      <td className="font-semibold text-gray-800">{user.name}</td>
                      <td className="text-gray-500">{user.email || '-'}</td>
                      <td>
                        {user.department ? (
       