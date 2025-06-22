"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Leaf } from "lucide-react"
import Link from "next/link"
import ImageUpload from "@/components/image-upload"

interface FreshnessResult {
  item_type: string
  freshness_score: number
  freshness_level: string
  recommendations: string[]
}

export default function FreshnessDetectionPage() {
  const [results, setResults] = useState<FreshnessResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)

  const analyzeImage = async (file: File) => {
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append("image", file)

      const response = await fetch("http://localhost:8000/api/freshness-detection", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to analyze image")
      }

      const data = await response.json()
      setResults(data.items || [])
    } catch (error) {
      console.error("Error analyzing image:", error)
      alert("Failed to analyze image. Please make sure the backend server is running.")
    } finally {
      setIsLoading(false)
    }
  }

  const getFreshnessColor = (level: string) => {
    switch (level.toLowerCase()) {
      case "fresh":
        return "bg-green-100 text-green-800"
      case "moderate":
        return "bg-yellow-100 text-yellow-800"
      case "poor":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-green-100">
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
            <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center text-white mr-4">
              <Leaf className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Freshness Detection</h1>
              <p className="text-gray-600">Analyze the freshness of fruits and vegetables</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Upload Image</CardTitle>
                <CardDescription>Upload an image of fruits or vegetables for freshness analysis</CardDescription>
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
                <CardTitle>Freshness Analysis</CardTitle>
                <CardDescription>Freshness levels and recommendations</CardDescription>
              </CardHeader>
              <CardContent>
                {results.length > 0 ? (
                  <div className="space-y-4">
                    {results.map((result, index) => (
                      <div key={index} className="p-4 border rounded-lg bg-white">
                        <div className="flex justify-between items-center mb-3">
                          <h3 className="font-semibold text-lg capitalize">{result.item_type}</h3>
                          <span
                            className={`text-sm px-3 py-1 rounded-full ${getFreshnessColor(result.freshness_level)}`}
                          >
                            {result.freshness_level}
                          </span>
                        </div>
                        <div className="mb-3">
                          <div className="flex justify-between text-sm mb-1">
                            <span>Freshness Score</span>
                            <span>{(result.freshness_score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-green-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${result.freshness_score * 100}%` }}
                            ></div>
                          </div>
                        </div>
                        {result.recommendations.length > 0 && (
                          <div>
                            <h4 className="font-medium text-sm mb-2">Recommendations:</h4>
                            <ul className="text-sm text-gray-600 space-y-1">
                              {result.recommendations.map((rec, recIndex) => (
                                <li key={recIndex} className="flex items-start">
                                  <span className="mr-2">•</span>
                                  <span>{rec}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <Leaf className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>Upload an image to analyze freshness</p>
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
