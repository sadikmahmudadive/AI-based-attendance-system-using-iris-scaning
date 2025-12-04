# Node.js Iris Attendance System

A modern web application for iris-based attendance tracking built with Node.js (Express) and React.

## Architecture

```
webapp/
├── server.js          # Express backend server
├── package.json       # Backend dependencies
├── client/            # React frontend
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   └── Navbar.js
│   │   └── pages/
│   │       ├── Dashboard.js
│   │       ├── MarkAttendance.js
│   │       ├── AttendanceReport.js
│   │       ├── EnrollUser.js
│   │       └── UserList.js
│   └── package.json
```

## Prerequisites

1. **Python Flask API** must be running (for iris recognition)
   ```
   python app.py
   ```
   This runs on `http://localhost:5000`

2. **Node.js** (v16 or higher)

## Installation

1. **Install backend dependencies:**
   ```bash
   cd webapp
   npm install
   ```

2. **Install frontend dependencies:**
   ```bash
   cd client
   npm install
   ```

## Running the Application

### Development Mode

1. **Start the Flask API** (in the main project folder):
   ```bash
   python app.py
   ```

2. **Start the Node.js server** (in webapp folder):
   ```bash
   npm run dev
   ```
   Runs on `http://localhost:5001`

3. **Start the React dev server** (in webapp/client folder):
   ```bash
   npm start
   ```
   Runs on `http://localhost:3000`

### Production Mode

1. **Build the React app:**
   ```bash
   cd client
   npm run build
   ```

2. **Start the server:**
   ```bash
   npm start
   ```
   
   Access at `http://localhost:5001`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/dashboard` | Dashboard statistics |
| POST | `/api/attendance/mark` | Mark attendance (upload image) |
| GET | `/api/attendance/records` | Get attendance records |
| GET | `/api/users` | List all enrolled users |
| POST | `/api/users/enroll` | Enroll new user |
| DELETE | `/api/users/:irisId` | Delete a user |
| POST | `/api/iris/detect` | Detect iris ID from image |

## Features

- 📊 **Dashboard** - Real-time attendance statistics
- 📸 **Mark Attendance** - Upload or webcam capture
- 📋 **Reports** - View and export attendance records
- 👤 **Enroll Users** - Register new users with iris detection
- 👥 **User Management** - View/delete enrolled users
- 🔄 **Real-time Updates** - Socket.io for live updates

## Tech Stack

**Backend:**
- Express.js
- Socket.io
- Multer (file uploads)
- Axios (Flask API communication)

**Frontend:**
- React 18
- React Router
- Tailwind CSS
- Lucide Icons
- React Hot Toast
- React Webcam
- React Dropzone
