"use client"

import React, { useState } from "react"; // Added React import
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { 
    ArrowLeft, 
    Calendar, 
    AlertTriangle, 
    CheckCircle, 
    Clock, 
    ScanText, 
    IndianRupee, 
    CalendarClock,
    Loader2 // Added Loader2 import
} from "lucide-react"; 
import Link from "next/link";
import ImageUpload from "@/components/image-upload"; // Ensure this path is correct

// Interface aligned with the backend's actual response for MRP/Expiry
interface MrpExpiryInfoFromApi {
  mrp: string;                     
  manufacturing_date_str: string;  
  best_before_str: string;         
  expiry_date_on_label_str: string; 
  parsed_mfg_date: string;         
  parsed_expiry_date_on_label: string; 
  calculated_expiry_date: string;  
  shelf_life_status: string;       
}


export default function MrpExpiryPage() {
  const [results, setResults] = useState<MrpExpiryInfoFromApi | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null); 


  const analyzeImage = async (file: File) => {
    setIsLoading(true);
    setResults(null);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch("http://localhost:8000/api/mrp-expiry", { 
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: `Server error: ${response.status}` }));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      if (data.success && data.mrp_expiry_info) {
        setResults(data.mrp_expiry_info);
      } else {
        throw new Error(data.message || "MRP and Expiry information not found in server response.");
      }
    } catch (error: any) {
      console.error("Error analyzing image:", error);
      setError(error.message || "Failed to analyze image. Please ensure the backend server is running.");
      setResults(null);
    } finally {
      setIsLoading(false);
    }
  };

  const getExpiryDisplayInfo = (statusText: string | undefined | null): {
    text: string;
    colorClass: string;
    Icon: React.ElementType;
   } => {
    const defaultInfo = { text: statusText || "N/A", colorClass: "text-gray-600 bg-gray-50 border-gray-200", Icon: Calendar };
    if (!statusText || statusText.toLowerCase() === "n/a") return defaultInfo;

    if (statusText.toLowerCase().includes("expired")) {
      return { text: statusText, colorClass: "text-red-600 bg-red-50 border-red-200", Icon: AlertTriangle };
    } else if (statusText.toLowerCase().includes("expires today")) {
      return { text: statusText, colorClass: "text-yellow-600 bg-yellow-50 border-yellow-200", Icon: Clock };
    } else if (statusText.toLowerCase().includes("expires in")) {
      if (statusText.toLowerCase().includes("days")) {
        const daysMatch = statusText.match(/(\d+)\s*day/);
        if (daysMatch && parseInt(daysMatch[1]) < 60) { 
            return { text: statusText, colorClass: "text-yellow-600 bg-yellow-50 border-yellow-200", Icon: Clock };
        }
      }
      return { text: statusText, colorClass: "text-green-600 bg-green-50 border-green-200", Icon: CheckCircle };
    }
    return defaultInfo;
  };


  const formatDate = (dateString: string | undefined | null) => {
    if (!dateString || dateString.toLowerCase() === "n/a") {
      return "N/A";
    }
    try {
      const dateObj = new Date(dateString);
      if (isNaN(dateObj.getTime())) { 
        return dateString; 
      }
      return dateObj.toLocaleDateString("en-IN", { 
        year: "numeric",
        month: "long",
        day: "numeric",
      });
    } catch {
      return dateString; 
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-purple-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Link href="/">
            <Button variant="ghost" size="sm" className="mr-4 hover:bg-purple-100">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Home
            </Button>
          </Link>
          <div className="flex items-center">
            <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center text-white mr-4">
              <ScanText className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">MRP & Expiry Detection</h1>
              <p className="text-gray-600">Extract pricing and expiry information from product labels</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>Upload Product Image</CardTitle>
                <CardDescription>Upload an image to detect MRP and Expiry Date</CardDescription>
              </CardHeader>
              <CardContent>
                <ImageUpload 
                  onImageSelect={(file) => setSelectedImage(file)} 
                  onAnalyze={analyzeImage} 
                  isLoading={isLoading} 
                />
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div>
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>Detected Information</CardTitle>
                <CardDescription>Extracted MRP and date details</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading && ( // Line 160 is around here if error was in this block
                  <div className="text-center py-8 flex flex-col items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-purple-600 mb-3" />
                    <p className="text-purple-600">Detecting MRP and Expiry Date...</p>
                  </div>
                )}
                {error && !isLoading && (
                  <div className="p-4 bg-red-100 text-red-700 rounded-lg text-center">
                    <AlertTriangle className="h-6 w-6 mx-auto mb-2 text-red-500" />
                    <p className="font-semibold">Error:</p>
                    <p>{error}</p>
                  </div>
                )}
                {!isLoading && !error && results && (
                  <div className="space-y-6">
                    <div className={`p-4 rounded-lg border-2 ${getExpiryDisplayInfo(results.shelf_life_status).colorClass} flex items-center shadow-sm`}>
                       {/* Line 174 is here */}
                      <div className="mr-3">{React.createElement(getExpiryDisplayInfo(results.shelf_life_status).Icon, { className: "h-6 w-6" })}</div>
                      <div>
                        <p className="font-semibold text-lg capitalize">
                            {results.shelf_life_status !== "N/A" ? results.shelf_life_status : "Shelf Life Status N/A"}
                        </p>
                        {results.calculated_expiry_date && results.calculated_expiry_date !== "N/A" && (
                             <p className="text-sm ">Expiry Date: {formatDate(results.calculated_expiry_date)}</p>
                        )}
                      </div>
                    </div>

                    <div className="p-4 bg-gray-50 rounded-lg shadow">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium text-gray-600 flex items-center"><IndianRupee className="h-4 w-4 mr-1" /> MRP</span>
                            <span className="text-xl font-bold text-purple-700">{results.mrp}</span>
                        </div>
                        <hr className="my-2"/>
                         <div className="flex justify-between items-center mt-2">
                            <span className="text-sm font-medium text-gray-600 flex items-center"><CalendarClock className="h-4 w-4 mr-1" /> Calculated Expiry</span>
                            <span className="font-semibold text-gray-800">{formatDate(results.calculated_expiry_date)}</span>
                        </div>
                    </div>
                    
                    <Card>
                        <CardHeader className="pb-2">
                            <CardTitle className="text-md">Raw Date Information</CardTitle>
                        </CardHeader>
                        <CardContent className="text-sm space-y-1 pt-0">
                            <div className="flex justify-between"><span className="text-gray-600">MFD (on label):</span> <span className="font-mono">{results.manufacturing_date_str}</span></div>
                            <div className="flex justify-between"><span className="text-gray-600">Best Before (on label):</span> <span className="font-mono">{results.best_before_str}</span></div>
                            <div className="flex justify-between"><span className="text-gray-600">Expiry (on label):</span> <span className="font-mono">{results.expiry_date_on_label_str}</span></div>
                             <hr className="my-1"/>
                            <div className="flex justify-between"><span className="text-gray-600">Parsed MFD:</span> <span className="font-mono">{formatDate(results.parsed_mfg_date)}</span></div>
                            <div className="flex justify-between"><span className="text-gray-600">Parsed Expiry (label):</span> <span className="font-mono">{formatDate(results.parsed_expiry_date_on_label)}</span></div>
                        </CardContent>
                    </Card>
                  </div>
                )}
                {!isLoading && !error && !results && (
                  <div className="text-center text-gray-500 py-8">
                    <ScanText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>Upload an image to see MRP and Expiry Date</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}