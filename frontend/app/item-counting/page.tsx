"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Hash } from "lucide-react"
import Link from "next/link"
import ImageUpload from "@/components/image-upload"

interface CountResult {
  item_type: string
  count: number
  confidence: number
  locations: number[][]
}

export default function ItemCountingPage() {
  const [results, setResults] = useState<CountResult[]>([])
  const [totalCount, setTotalCount] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)

  const analyzeImage = async (file: File) => {
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append("image", file)

      const response = await fetch("http://localhost:8000/api/item-counting", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to analyze image")
      }

      const data = await response.json()
      setResults(data.items || [])
      setTotalCount(data.total_count || 0)
    } catch (error) {
      console.error("Error analyzing image:", error)
      alert("Failed to analyze image. Please make sure the backend server is running.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-purple-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Link href="/">
            <Button variant="ghost" size="sm" className="mr-4">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Home
            </Button>
          </Link>
          <div className="flex items-center">
            <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center text-white mr-4">
              <Hash className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Item Counting</h1>
              <p className="text-gray-600">Automatically count items in warehouse images</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Upload Image</CardTitle>
                <CardDescription>Upload an image containing items to count</CardDescription>
              </CardHeader>
              <CardContent>
                <ImageUpload onImageSelect={setSelectedImage} onAnalyze={analyzeImage} isLoading={isLoading} />
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Counting Results</CardTitle>
                <CardDescription>Detected items and their counts</CardDescription>
              </CardHeader>
              <CardContent>
                {results.length > 0 ? (
                  <div className="space-y-4">
                    {/* Total Count */}
                    <div className="p-4 bg-purple-50 rounded-lg border-2 border-purple-200">
                      <div className="text-center">
                        <h3 className="text-2xl font-bold text-purple-800">Total Items</h3>
                        <p className="text-4xl font-bold text-purple-600">{totalCount}</p>
                      </div>
                    </div>

                    {/* Individual Items */}
                    {results.map((result, index) => (
                      <div key={index} className="p-4 border rounded-lg bg-white">
                        <div className="flex justify-between items-center mb-2">
                          <h3 className="font-semibold text-lg capitalize">{result.item_type}</h3>
                          <div className="flex items-center gap-2">
                            <span className="text-2xl font-bold text-purple-600">{result.count}</span>
                            <span className="text-sm bg-purple-100 text-purple-800 px-2 py-1 rounded">
                              {(result.confidence * 100).toFixed(1)}% confidence
                            </span>
                          </div>
                        </div>
                        <div className="text-sm text-gray-600">Detected {result.locations.length} instances</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <Hash className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>Upload an image to count items</p>
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
