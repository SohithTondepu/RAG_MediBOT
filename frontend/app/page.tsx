"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { medicalApi } from "@/lib/api";
import { Message } from "@/types";
import {
  Send,
  Upload,
  FileText,
  Loader2,
  Stethoscope,
  BookOpen,
  Menu,
  X,
  Trash2,
  ExternalLink,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

import dynamic from "next/dynamic";
import { Suspense } from "react";

const PDFViewer = dynamic(() => import("@/components/PDFViewer"), {
  ssr: false,
  loading: () => (
    <div className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center">
      <div className="bg-white p-6 rounded-lg">
        <Loader2 className="w-8 h-8 text-[#0B4F6C] animate-spin" />
      </div>
    </div>
  ),
});

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "👋 Welcome to **MediBot** - Your Medical Assistant",
      timestamp: new Date(),
    },
  ]);

  const [pdfViewer, setPdfViewer] = useState<{
    isOpen: boolean;
    filePath: string;
    pageNumber: number;
  }>({
    isOpen: false,
    filePath: "",
    pageNumber: 1,
  });

  

  const handleSourceClick = (source: string) => {
    // Extract page number from source string (e.g., "page: 5" -> 5)
    const pageMatch = source.match(/page:?\s*(\d+)/i);
    const pageNumber = pageMatch ? parseInt(pageMatch[1]) : 1;

    // Use the active document name if available, otherwise fallback to Medical_book.pdf
    const fileName = activeDocName || "Medical_book.pdf";
    

    setPdfViewer({
      isOpen: true,
      filePath: `http://localhost:8000/upload/${fileName}`,
      pageNumber: pageNumber,
    });
  };

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeDocName, setActiveDocName] = useState<string | null>(null);
  const [showSources, setShowSources] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isDragActive, setIsDragActive] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);



  const timeFormatter = useMemo(
    () =>
      new Intl.DateTimeFormat(undefined, {
        hour: "2-digit",
        minute: "2-digit",
      }),
    [],
  );

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Close sidebar on Escape (mobile UX)
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsSidebarOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // Lock page scroll (so only sidebar content/chat messages scroll)
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  const resetFileInput = () => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type !== "application/pdf") {
        alert("Please upload only PDF files");
        return;
      }
      setSelectedFile(file);
    }
  };

  // Drag & drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      alert("Please upload only PDF files");
      return;
    }

    setSelectedFile(file);
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);

    try {
      const result = await medicalApi.uploadPDF(selectedFile);

      setActiveDocName(selectedFile.name);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: `✅ Successfully uploaded "${selectedFile.name}"\n\n📄 Created ${result.chunks_created} document chunks. You can now ask questions about this medical document.`,
          timestamp: new Date(),
        },
      ]);

      resetFileInput();
      setIsSidebarOpen(false);

      requestAnimationFrame(() => textareaRef.current?.focus());
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: `❌ Failed to upload "${selectedFile.name}". Please try again.`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle sending message
  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const content = input.trim();

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    requestAnimationFrame(() => textareaRef.current?.focus());

    try {
      const response = await medicalApi.askQuestion(content);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content:
            "I apologize, but I encountered an error. Please make sure the backend server is running.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* IMPORTANT: fixed viewport height + overflow hidden => page does not scroll */}
      <div className="relative flex h-dvh overflow-hidden bg-linear-to-b from-[#F6FAFD] to-[#EEF6FB]">
        {/* Mobile Header */}
        <div className="lg:hidden fixed top-0 left-0 right-0 z-40 bg-white/90 backdrop-blur border-b border-[#D9E9F2] px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-xl bg-[#E6F4FB] flex items-center justify-center">
              <Stethoscope className="w-5 h-5 text-[#0B4F6C]" />
            </div>
            <div className="leading-tight">
              <h1 className="text-base font-bold text-[#0B4F6C]">MediBot</h1>
              <p className="text-[11px] text-[#2C7DA0]">
                Medical Document Assistant
              </p>
            </div>
          </div>

          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 rounded-xl hover:bg-[#F0F4F8] active:bg-[#E6F4FB] transition-colors focus:outline-none focus:ring-2 focus:ring-[#61A5C2]"
            aria-label={isSidebarOpen ? "Close menu" : "Open menu"}
          >
            {isSidebarOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Overlay for mobile when sidebar is open */}
        {isSidebarOpen && (
          <div
            className="fixed inset-0 bg-black/40 z-30 lg:hidden"
            onClick={() => setIsSidebarOpen(false)}
            aria-hidden="true"
          />
        )}

        {/* Sidebar */}
        <aside
          className={[
            "fixed lg:static inset-y-0 left-0 z-40 lg:z-auto",
            "w-[18rem] sm:w-80",
            "bg-white/95 backdrop-blur",
            "border-r border-[#D9E9F2]",
            "flex flex-col min-h-0", // <-- allows inner scroll area to work without scrolling page
            "transform transition-transform duration-300 ease-out",
            isSidebarOpen
              ? "translate-x-0"
              : "-translate-x-full lg:translate-x-0",
            "shadow-2xl lg:shadow-none",
          ].join(" ")}
          aria-label="Sidebar"
        >
          <div className="px-5 py-5 border-b border-[#D9E9F2] pt-20 lg:pt-6">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-11 h-11 rounded-2xl bg-[#E6F4FB] flex items-center justify-center shrink-0">
                  <Stethoscope className="w-6 h-6 text-[#0B4F6C]" />
                </div>
                <div className="min-w-0">
                  <h1 className="text-xl font-bold text-[#0B4F6C] truncate">
                    MediBot
                  </h1>
                  <p className="text-xs text-[#2C7DA0]">
                    Medical Document Assistant
                  </p>
                </div>
              </div>

              <button
                className="lg:hidden p-2 rounded-xl hover:bg-[#F0F4F8] transition-colors focus:outline-none focus:ring-2 focus:ring-[#61A5C2]"
                onClick={() => setIsSidebarOpen(false)}
                aria-label="Close sidebar"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {activeDocName && (
              <div className="mt-4 rounded-xl bg-[#F0F8FD] border border-[#D9E9F2] px-3 py-2">
                <div className="flex items-start gap-2">
                  <FileText className="w-4 h-4 text-[#0B4F6C] mt-0.5 shrink-0" />
                  <div className="min-w-0">
                    <p className="text-[11px] text-[#2C7DA0]">Active document</p>
                    <p className="text-sm text-[#0B1A2B] font-medium truncate">
                      {activeDocName}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar scroll area (only this scrolls inside sidebar) */}
          <div className="p-5 flex-1 min-h-0 overflow-y-auto">
            <div className="rounded-2xl p-4 border border-[#D9E9F2] bg-white shadow-sm">
              <div className="flex items-center gap-2 text-[#0B4F6C] mb-1">
                <BookOpen className="w-4 h-4" />
                <h2 className="font-semibold">Upload Medical PDF</h2>
              </div>
              <p className="text-xs text-[#2C7DA0] mb-3">
                Drag & drop a PDF, or choose one to start.
              </p>

              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept=".pdf"
                className="hidden"
                id="pdf-upload"
              />

              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={[
                  "rounded-2xl border-2 border-dashed p-4 transition-colors",
                  isDragActive
                    ? "border-[#0B4F6C] bg-[#E6F4FB]"
                    : "border-[#61A5C2] bg-[#F7FCFF]",
                ].join(" ")}
              >
                {!selectedFile ? (
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full py-3 px-4 bg-white rounded-xl border border-[#D9E9F2] text-[#0B4F6C] hover:bg-[#F0F4F8] transition-colors flex items-center justify-center gap-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#61A5C2]"
                    disabled={isUploading}
                  >
                    <Upload className="w-4 h-4" />
                    <span>Select PDF</span>
                  </button>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 bg-white p-3 rounded-xl border border-[#D9E9F2]">
                      <FileText className="w-4 h-4 text-[#0B4F6C] shrink-0" />
                      <span className="text-sm truncate flex-1 text-[#0B1A2B]">
                        {selectedFile.name}
                      </span>
                      <button
                        type="button"
                        onClick={resetFileInput}
                        className="p-2 rounded-lg hover:bg-[#F0F4F8] transition-colors focus:outline-none focus:ring-2 focus:ring-[#61A5C2]"
                        aria-label="Remove selected file"
                        disabled={isUploading}
                      >
                        <Trash2 className="w-4 h-4 text-[#2C7DA0]" />
                      </button>
                    </div>

                    <button
                      onClick={handleUpload}
                      disabled={isUploading}
                      className="w-full py-2.5 bg-[#0B4F6C] text-white rounded-xl hover:bg-[#2C7DA0] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm focus:outline-none focus:ring-2 focus:ring-[#61A5C2]"
                    >
                      {isUploading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Processing…</span>
                        </>
                      ) : (
                        <span>Upload to MediBot</span>
                      )}
                    </button>
                  </div>
                )}

                
              </div>
            </div>

            <div className="mt-5 rounded-2xl bg-[#F0F4F8] border border-[#D9E9F2] p-4">
              <p className="text-xs font-semibold text-[#0B4F6C] mb-2">Tips</p>
              <ul className="text-xs text-[#2C7DA0] space-y-1 list-disc pl-4">
                <li>Ask for summaries, abnormal values, or key findings.</li>
                <li>Try: “Explain this in simple terms.”</li>
                <li>Use sources to verify where answers came from.</li>
              </ul>
            </div>
          </div>
        </aside>

        {/* Main Chat Area */}
        <main className="flex-1 min-w-0 min-h-0 flex flex-col w-full pt-14 lg:pt-0">
          {/* Desktop header */}
          <div className="hidden lg:flex items-center justify-between bg-white/90 backdrop-blur border-b border-[#D9E9F2] px-6 py-4 sticky top-0 z-10">
            <div>
              <h2 className="text-lg font-semibold text-[#0B4F6C]">
                Medical Consultation
              </h2>
              <p className="text-xs text-[#2C7DA0]">
                {activeDocName
                  ? `Working with: ${activeDocName}`
                  : "Upload a PDF to begin"}
              </p>
            </div>
          </div>

          {/* Messages (ONLY THIS SCROLLS) */}
          <div className="flex-1 min-h-0 overflow-y-auto px-4 lg:px-6 py-5">
            <div className="max-w-3xl mx-auto space-y-4">
              {messages.map((message) => {
                const isUser = message.role === "user";

                return (
                  <div
                    key={message.id}
                    className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={[
                        "max-w-[92%] sm:max-w-[80%]",
                        "rounded-2xl px-4 py-3 lg:px-5 lg:py-4",
                        "shadow-sm",
                        isUser
                          ? "bg-[#0B4F6C] text-white rounded-br-md"
                          : "bg-white text-[#0B1A2B] rounded-bl-md border border-[#E6F4FB]",
                      ].join(" ")}
                    >
                      <div
                        className={[
                          "prose prose-sm lg:prose-base max-w-none",
                          "prose-p:my-2 prose-li:my-1 prose-ul:my-2",
                          isUser ? "prose-invert" : "",
                        ].join(" ")}
                      >
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>

                      {message.sources && message.sources.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-[#D9E9F2]/80">
                          <button
                            onClick={() =>
                              setShowSources(
                                showSources === message.id ? null : message.id,
                              )
                            }
                            className={[
                              "text-xs inline-flex items-center gap-1.5",
                              isUser
                                ? "text-white/80 hover:text-white"
                                : "text-[#2C7DA0] hover:text-[#0B4F6C]",
                              "transition-colors",
                              "focus:outline-none focus:ring-2 focus:ring-[#61A5C2] rounded-md px-1",
                            ].join(" ")}
                            aria-expanded={showSources === message.id}
                          >
                            <BookOpen className="w-3.5 h-3.5" />
                            <span>Sources ({message.sources.length})</span>
                          </button>

                          {showSources === message.id && (
                            <div className="mt-2 space-y-2">
                              {message.sources.map((source, idx) => {
                                const pageMatch = source.match(/page:?\s*(\d+)/i);
                                const pageNumber = pageMatch ? parseInt(pageMatch[1]) : null;
                                
                                return (
                                  <div
                                    key={idx}
                                    className={[
                                      "text-xs p-2.5 rounded-xl wrap-break-word",
                                      "flex items-start justify-between gap-2",
                                      isUser
                                        ? "bg-white/10 text-white/90"
                                        : "bg-[#F0F4F8] text-[#0B1A2B]",
                                    ].join(" ")}
                                  >
                                    <span className="flex-1">📄 {source}</span>
                                    {pageNumber && (
                                      <button
                                        onClick={() => handleSourceClick(source)}
                                        className={[
                                          "inline-flex items-center gap-1 px-2 py-1 rounded-lg text-xs",
                                          "hover:bg-[#0B4F6C] hover:text-white",
                                          "transition-colors focus:outline-none focus:ring-2 focus:ring-[#61A5C2]",
                                          isUser
                                            ? "bg-white/20 text-white"
                                            : "bg-white text-[#0B4F6C] border border-[#D9E9F2]",
                                        ].join(" ")}
                                        title={`View page ${pageNumber}`}
                                      >
                                        <ExternalLink className="w-3 h-3" />
                                        <span>View</span>
                                      </button>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}

                      <span
                      suppressHydrationWarning
                        className={`text-[11px] mt-2 block ${isUser ? "text-white/70" : "text-[#2C7DA0]"}`}
                      >
                        {timeFormatter.format(message.timestamp)}
                      </span>
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-2xl rounded-bl-md px-5 py-4 shadow-sm border border-[#E6F4FB]">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 bg-[#2C7DA0] rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      />
                      <div
                        className="w-2 h-2 bg-[#2C7DA0] rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      />
                      <div
                        className="w-2 h-2 bg-[#2C7DA0] rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      />
                      <span className="ml-2 text-xs text-[#2C7DA0]">
                        Thinking…
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area (fixed in layout; does not scroll with messages) */}
          <div className="bg-white/95 backdrop-blur border-t border-[#D9E9F2] px-3 lg:px-4 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))]">
            <div className="max-w-3xl mx-auto flex items-end gap-2 lg:gap-3">
              <div className="flex-1">
                <label htmlFor="chat-input" className="sr-only">
                  Ask about your medical document
                </label>
                <textarea
                  id="chat-input"
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder={
                    activeDocName
                      ? "Ask about your medical document"
                      : "Upload a PDF, then ask questions…"
                  }
                  className="w-full resize-none min-h-11 max-h-32 leading-5 p-3 border border-[#D9E9F2] rounded-2xl focus:outline-none focus:ring-2 focus:ring-[#61A5C2] text-sm lg:text-base bg-white shadow-sm"
                  disabled={isLoading || isUploading}
                  rows={1}
                />
              </div>

              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || isLoading || isUploading}
                className="h-11 w-11 lg:h-12 lg:w-12 bg-[#0B4F6C] text-white rounded-2xl hover:bg-[#2C7DA0] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-sm focus:outline-none focus:ring-2 focus:ring-[#61A5C2]"
                aria-label="Send message"
              >
                <Send className="w-4 h-4 lg:w-5 lg:h-5" />
              </button>
            </div>
          </div>
        </main>
      </div>

      {/* PDF Viewer Modal */}
      {pdfViewer.isOpen && (
        <Suspense fallback={null}>
          <PDFViewer
            file={pdfViewer.filePath}
            pageNumber={pdfViewer.pageNumber}
            onClose={() => setPdfViewer(prev => ({ ...prev, isOpen: false }))}
          />
        </Suspense>
      )}
    </>
  );
}