import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch('http://localhost:8000/');
    const data = await response.json();
    return NextResponse.json({ 
      success: true, 
      backend: data,
      message: "Backend is reachable!" 
    });
  } catch (error) {
    return NextResponse.json({ 
      success: false, 
      error: "Cannot connect to backend. Make sure it's running on port 8000" 
    }, { status: 500 });
  }
}