// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getDatabase } from "firebase/database";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCCg0SSEETzEQ6zmnAHZeRrabKIGWWYhx4",
  authDomain: "rational-cat-473121-k4.firebaseapp.com",
  databaseURL: "https://rational-cat-473121-k4-default-rtdb.firebaseio.com",
  projectId: "rational-cat-473121-k4",
  storageBucket: "rational-cat-473121-k4.firebasestorage.app",
  messagingSenderId: "935672815123",
  appId: "1:935672815123:web:acc0bc1c1e37abd31ee9e6",
  measurementId: "G-LFJGXNHJCC"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const database = getDatabase(app);

// Export Firebase instances for use in other modules
export { app, analytics, database };
export default app;
