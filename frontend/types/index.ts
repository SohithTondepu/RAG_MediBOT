export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: string[];  // ["page: 1", "page: 5"]
};

export type QueryRequest = {
  question: string;
};

export type QueryResponse = {
  answer: string;
  sources: string[] | null;
};

export type UploadResponse = {
  message: string;
  chunks_created: number;
};

// Add this for PDF viewer
export type PDFViewerProps = {
  isOpen: boolean;
  filePath: string;
  pageNumber: number;
  onClose: () => void;
};