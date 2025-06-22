"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Eye, Package, Tag, ShoppingBag, Hash, Percent } from "lucide-react" // Added more icons
import Link from "next/link"
import ImageUpload from "@/components/image-upload" // Ensure this path is correct

// Updated interface to match the new backend response
interface BrandProductInfo {
  brand_name: string;
  product_name: string;
  variant_description: string;
  quantity: number;
  confidence: number;
}

export default function BrandRecognitionPage() {
  const [results, setResults] = useState<BrandProductInfo[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null) // Used by ImageUpload
  const [error, setError] = useState<string | null>(null);


  const analyzeImage = async (file: File) => {
    setIsLoading(true)
    setResults([]) // Clear previous results
    setError(null);
    try {
      const formData = new FormData()
      formData.append("image", file)

      const response = await fetch("http://localhost:8000/api/brand-recognition", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to analyze image. Server response not OK." }));
        throw new Error(errorData.detail || `Failed to analyze image: ${response.statusText}`);
      }

      const data = await response.json()
      if (data.success && data.brands) {
        setResults(data.brands)
      } else {
        setResults([])
        // Consider if data.message should be shown as an error or info
        // For now, if brands array is missing or empty, we'll show the default "no results" message.
        if (!data.brands || data.brands.length === 0) {
            console.log("No brands detected or returned by backend.");
        }
      }
    } catch (err: any) {
      console.error("Error analyzing image for brand recognition:", err)
      setError(err.message || "An unexpected error occurred. Please ensure the backend is running.");
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-blue-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Link href="/">
            <Button variant="ghost" size="sm" className="mr-4 hover:bg-blue-100">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Home
            </Button>
          </Link>
          <div className="flex items-center">
            <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center text-white mr-4">
              <ShoppingBag className="h-6 w-6" /> {/* Changed Icon */}
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Brand & Product Recognition</h1>
              <p className="text-gray-600">Identify brands, products, variants, and quantities in images</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>Upload Image</CardTitle>
                <CardDescription>Upload an image containing products for analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <ImageUpload 
                    onImageSelect={setSelectedImage} 
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
                <CardTitle>Recognition Results</CardTitle>
                <CardDescription>Detected brands, products, and quantities</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading && (
                    <div className="text-center py-8">
                        <p className="text-blue-600">Analyzing image for brands and products...</p>
                    </div>
                )}
                {error && !isLoading && (
                  <div className="p-4 bg-red-100 text-red-700 rounded-lg text-center">
                    <p>Error: {error}</p>
                  </div>
                )}
                {!isLoading && !error && results.length > 0 && (
                  <div className="space-y-4">
                    {results.map((result, index) => (
                      <div key={index} className="p-4 border border-gray-200 rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <h3 className="font-semibold text-lg text-blue-700 flex items-center">
                                <Tag className="h-5 w-5 mr-2 text-blue-500"/> {result.brand_name}
                            </h3>
                            <p className="text-md text-gray-800 flex items-center ml-1">
                                <Package className="h-4 w-4 mr-2 text-gray-500"/> {result.product_name}
                            </p>
                          </div>
                          <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded-full flex items-center">
                            <Percent className="h-3 w-3 mr-1"/> {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <p className="flex items-center"><ShoppingBag className="h-4 w-4 mr-2 text-gray-400"/> Variant: {result.variant_description}</p>
                          <p className="flex items-center"><Hash className="h-4 w-4 mr-2 text-gray-400"/> Quantity: {result.quantity}</p>
                          {/* Bounding box display removed as it's not in the new data */}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {!isLoading && !error && results.length === 0 && (
                  <div className="text-center text-gray-500 py-8">
                    <Eye className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>Upload an image to see brand and product recognition results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}