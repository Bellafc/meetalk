// axiosConfig.js

import axios from 'axios';

// Create a new instance of Axios with a custom configuration
const axiosInstance = axios.create({
  baseURL: 'https://www.gdcasa.cn:3007/api', // Replace 'https://api.example.com' with your base URL
});

export default axiosInstance;
