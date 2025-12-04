/**
 * Iris Attendance System - Node.js Backend
 * =========================================
 * Express server that communicates with the Python Flask API
 * for iris recognition and provides a modern REST API.
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { Server } = require('socket.io');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "http://localhost:3000",
        methods: ["GET", "POST"]
    }
});

// Configuration
const PORT = process.env.PORT || 5001;
const FLASK_API_URL = process.env.FLASK_API_URL || 'http://localhost:5000';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from React build
app.use(express.static(path.join(__dirname, 'client/build')));

// File upload configuration
const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only JPEG, PNG, BMP, TIFF allowed.'));
        }
    }
});

// Data storage paths
const DATA_DIR = path.join(__dirname, '../models');
const ENROLLMENT_DB = path.join(DATA_DIR, 'enrollment_database.json');
const ATTENDANCE_LOGS_DIR = path.join(DATA_DIR, 'attendance_logs');

// Ensure directories exist
if (!fs.existsSync(ATTENDANCE_LOGS_DIR)) {
    fs.mkdirSync(ATTENDANCE_LOGS_DIR, { recursive: true });
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function loadEnrollmentDB() {
    try {
        if (fs.existsSync(ENROLLMENT_DB)) {
            return JSON.parse(fs.readFileSync(ENROLLMENT_DB, 'utf8'));
        }
    } catch (error) {
        console.error('Error loading enrollment DB:', error);
    }
    return { enrollments: {}, metadata: { created: new Date().toISOString() } };
}

function saveEnrollmentDB(data) {
    data.metadata.last_updated = new Date().toISOString();
    fs.writeFileSync(ENROLLMENT_DB, JSON.stringify(data, null, 2));
}

function getTodayLogPath() {
    const today = new Date().toISOString().split('T')[0];
    return path.join(ATTENDANCE_LOGS_DIR, `attendance_${today}.csv`);
}

function getAttendanceRecords(date) {
    const logPath = path.join(ATTENDANCE_LOGS_DIR, `attendance_${date}.csv`);
    if (!fs.existsSync(logPath)) return [];
    
    const content = fs.readFileSync(logPath, 'utf8');
    const lines = content.trim().split('\n');
    if (lines.length <= 1) return [];
    
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
        const values = line.split(',');
        const record = {};
        headers.forEach((h, i) => record[h] = values[i]);
        return record;
    });
}

function getAttendanceSummary(date) {
    const records = getAttendanceRecords(date);
    const checkIns = records.filter(r => r.entry_type === 'CHECK_IN');
    const uniquePeople = new Set(checkIns.map(r => r.iris_id));
    
    return {
        total: uniquePeople.size,
        on_time: new Set(checkIns.filter(r => r.status === 'ON_TIME').map(r => r.iris_id)).size,
        late: new Set(checkIns.filter(r => r.status === 'LATE').map(r => r.iris_id)).size,
        early: new Set(checkIns.filter(r => r.status === 'EARLY').map(r => r.iris_id)).size,
        total_entries: records.length
    };
}

// ============================================================================
// API ROUTES
// ============================================================================

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Iris Attendance System API' });
});

// Get dashboard summary
app.get('/api/dashboard', (req, res) => {
    try {
        const today = new Date().toISOString().split('T')[0];
        const summary = getAttendanceSummary(today);
        const db = loadEnrollmentDB();
        const enrolledCount = Object.keys(db.enrollments).length;
        
        res.json({
            success: true,
            data: {
                date: today,
                summary,
                enrolled_count: enrolledCount
            }
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Mark attendance via iris recognition
app.post('/api/attendance/mark', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, error: 'No image provided' });
        }

        // Forward to Flask API for recognition
        const formData = new FormData();
        formData.append('image', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post(`${FLASK_API_URL}/api/recognize`, formData, {
            headers: formData.getHeaders()
        });

        const result = response.data;
        
        // Emit real-time update
        io.emit('attendance_update', result);
        
        res.json({ success: true, data: result });
    } catch (error) {
        console.error('Recognition error:', error.response?.data || error.message);
        res.status(500).json({ 
            success: false, 
            error: error.response?.data?.error || 'Recognition failed' 
        });
    }
});

// Get attendance records
app.get('/api/attendance/records', (req, res) => {
    try {
        const date = req.query.date || new Date().toISOString().split('T')[0];
        const records = getAttendanceRecords(date);
        const summary = getAttendanceSummary(date);
        
        res.json({
            success: true,
            data: { date, records, summary }
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get all enrolled users
app.get('/api/users', (req, res) => {
    try {
        const db = loadEnrollmentDB();
        const users = Object.entries(db.enrollments).map(([id, data]) => ({
            iris_id: id,
            ...data
        }));
        
        res.json({ success: true, data: users });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Enroll a new user
app.post('/api/users/enroll', (req, res) => {
    try {
        const { iris_id, name, email, department, employee_id, phone } = req.body;
        
        if (!iris_id || !name) {
            return res.status(400).json({ 
                success: false, 
                error: 'Iris ID and Name are required' 
            });
        }
        
        const db = loadEnrollmentDB();
        
        if (db.enrollments[iris_id]) {
            return res.status(400).json({ 
                success: false, 
                error: 'Iris ID already enrolled' 
            });
        }
        
        db.enrollments[iris_id] = {
            name,
            email: email || '',
            department: department || '',
            employee_id: employee_id || '',
            phone: phone || '',
            enrolled_at: new Date().toISOString(),
            is_active: true
        };
        
        saveEnrollmentDB(db);
        
        io.emit('user_enrolled', { iris_id, name });
        
        res.json({ 
            success: true, 
            message: `Successfully enrolled ${name}`,
            data: db.enrollments[iris_id]
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Delete a user
app.delete('/api/users/:irisId', (req, res) => {
    try {
        const { irisId } = req.params;
        const db = loadEnrollmentDB();
        
        if (!db.enrollments[irisId]) {
            return res.status(404).json({ 
                success: false, 
                error: 'User not found' 
            });
        }
        
        const userName = db.enrollments[irisId].name;
        delete db.enrollments[irisId];
        saveEnrollmentDB(db);
        
        io.emit('user_deleted', { iris_id: irisId, name: userName });
        
        res.json({ 
            success: true, 
            message: `Successfully deleted ${userName}` 
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get attendance summary for date range
app.get('/api/attendance/summary', (req, res) => {
    try {
        const { start_date, end_date } = req.query;
        const today = new Date().toISOString().split('T')[0];
        
        const startDate = start_date || today;
        const endDate = end_date || today;
        
        const summary = getAttendanceSummary(startDate);
        
        res.json({
            success: true,
            data: { start_date: startDate, end_date: endDate, summary }
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Detect iris from image (returns iris ID without marking attendance)
app.post('/api/iris/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ success: false, error: 'No image provided' });
        }

        const formData = new FormData();
        formData.append('image', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // Use a detection-only endpoint or parse recognition result
        const response = await axios.post(`${FLASK_API_URL}/api/recognize`, formData, {
            headers: formData.getHeaders()
        });

        const result = response.data;
        
        res.json({ 
            success: true, 
            data: {
                iris_id: result.iris_id,
                confidence: result.confidence,
                top_k: result.top_k
            }
        });
    } catch (error) {
        res.status(500).json({ 
            success: false, 
            error: error.response?.data?.error || 'Detection failed' 
        });
    }
});

// Socket.io connection handling
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
    });
});

// Serve React app for any other routes
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'client/build', 'index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err.message);
    res.status(500).json({ success: false, error: err.message });
});

// Start server
server.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('🎯 IRIS ATTENDANCE SYSTEM - Node.js Server');
    console.log('='.repeat(60));
    console.log(`   Server running at: http://localhost:${PORT}`);
    console.log(`   Flask API at: ${FLASK_API_URL}`);
    console.log('='.repeat(60) + '\n');
});
