'use client';

import { X } from 'lucide-react';

interface SimplePDFViewerProps {
  file: string;
  pageNumber: number;
  onClose: () => void;
}

export default function SimplePDFViewer({ file, pageNumber, onClose }: SimplePDFViewerProps) {
  // Add #page=xx for page navigation
  const pdfUrl = `${file}#page=${pageNumber}`;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg w-full max-w-5xl h-[90vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b">
          <h3 className="text-lg font-semibold text-[#0B4F6C]">
            Medical Document - Page {pageNumber}
          </h3>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 bg-gray-100">
          <iframe
            src={pdfUrl}
            className="w-full h-full"
            title="PDF Viewer"
          />
        </div>
      </div>
    </div>
  );
}