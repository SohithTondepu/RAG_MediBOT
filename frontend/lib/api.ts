import axios from "axios";
import { QueryResponse, UploadResponse } from "@/types";

const API_BASE_URL = "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const medicalApi = {
  // Test connection
  testConnection: async () => {
    try {
      const response = await api.get("/");
      return response.data;
    } catch (error) {
      throw new Error("Cannot connect to backend server");
    }
  },

  // Ask question - matches your /ask endpoint
  askQuestion: async (question: string): Promise<QueryResponse> => {
    try {
      const response = await api.post("/ask/", {
        question, // Your backend expects { "question": "..." }
      });

      return {
        answer: response.data.answer,
        sources: response.data.sources || [],
      };
    } catch (error) {
      console.error("API Error:", error);
      throw error;
    }
  },

  uploadPDF: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append("file", file); // Your backend expects 'file' field

    const response = await api.post("/upload/", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  // Add this to your medicalApi object
  checkFileExists: async (filename: string): Promise<boolean> => {
    try {
      const response = await api.get(`/check-file/${filename}`);
      return response.data;
    } catch (error) {
      return false;
    }
  },
};
