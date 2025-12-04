import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import MarkAttendance from './pages/MarkAttendance';
import AttendanceReport from './pages/AttendanceReport';
import EnrollUser from './pages/EnrollUser';
import UserList from './pages/UserList';

function App() {
  return (
    <Router>
      <div className="min-h-screen">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/mark-attendance" element={<MarkAttendance />} />
            <Route path="/reports" element={<AttendanceReport />} />
            <Route path="/enroll" element={<EnrollUser />} />
            <Route path="/users" element={<UserList />} />
          </Routes>
        </main>
        <footer className="text-center text-white py-4 mt-8">
          <p>&copy; 2024 Iris Attendance System | Powered by AI</p>
        </footer>
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1e293b',
              color: '#fff',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
