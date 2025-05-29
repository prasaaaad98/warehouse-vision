"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Zap } from "lucide-react"
import Link from "next/link"
import ImageUpload from "@/components/image-upload" // Assuming this path is correct

// Define the structure for a single nutrient item in the array
interface NutrientDetail {
  nutrient_name: string;
  amount: string;
  daily_value?: string;
}

interface NutritionInfo {
  product_name: string;
  serving_size: string;
  calories: number;
  nutrients: NutrientDetail[]; // Changed to an array of NutrientDetail
  ingredients: string[]; // Assuming backend sends 'ingredients' not 'ingredients_list'
  allergens: string[];   // Assuming backend sends 'allergens' not 'allergens_list'
}

export default function NutritionInfoPage() {
  const [results, setResults] = useState<NutritionInfo | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null) // This state seems unused here, ImageUpload handles its own image

  const analyzeImage = async (file: File) => {
    setIsLoading(true)
    setResults(null); // Clear previous results
    try {
      const formData = new FormData()
      formData.append("image", file)

      const response = await fetch("http://localhost:8000/api/nutrition-info", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to analyze image. Server response not OK." }));
        throw new Error(errorData.detail || "Failed to analyze image");
      }

      const data = await response.json();
      // The backend sends the nutrition data within a `nutrition_info` key
      // And the backend's `gemini_nutrition_extraction` uses ingredients_list and allergens_list
      // Let's map them if necessary or ensure consistency
      const rawNutritionInfo = data.nutrition_info;
      if (rawNutritionInfo) {
        setResults({
            product_name: rawNutritionInfo.product_name || "N/A",
            serving_size: rawNutritionInfo.serving_size || "N/A",
            calories: rawNutritionInfo.calories || 0,
            nutrients: rawNutritionInfo.nutrients || [], // Expecting an array
            ingredients: rawNutritionInfo.ingredients_list || [], // Map from backend key
            allergens: rawNutritionInfo.allergens_list || []     // Map from backend key
        });
      } else {
        setResults(null);
        throw new Error("Nutrition information not found in server response.");
      }

    } catch (error: any) {
      console.error("Error analyzing image:", error)
      alert(error.message || "Failed to analyze image. Please make sure the backend server is running.")
      setResults(null); // Clear results on error
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-orange-100">
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
            <div className="w-12 h-12 bg-orange-500 rounded-lg flex items-center justify-center text-white mr-4">
              <Zap className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Nutrition Info</h1>
              <p className="text-gray-600">Extract nutritional information from food labels</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Upload Image</CardTitle>
                <CardDescription>Upload an image of a nutrition label for analysis</CardDescription>
              </CardHeader>
              <CardContent>
                {/* Pass setSelectedImage if ImageUpload needs to lift its state up, 
                  otherwise ImageUpload calls onAnalyze directly with its internally managed file.
                  The current ImageUpload calls onAnalyze(selectedImageFromItsOwnState)
                */}
                <ImageUpload 
                  onImageSelect={(file) => setSelectedImage(file)} // This updates the parent's selectedImage state
                  onAnalyze={analyzeImage} // This analyzeImage will be called by ImageUpload
                  isLoading={isLoading} 
                />
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          <div>
            <Card>
              <CardHeader>
                <CardTitle>Nutrition Information</CardTitle>
                <CardDescription>Extracted nutritional data and ingredients</CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading && (
                    <div className="text-center text-gray-500 py-8">
                        <p>Loading nutrition information...</p>
                    </div>
                )}
                {!isLoading && results ? (
                  <div className="space-y-6">
                    {/* Product Info */}
                    <div className="p-4 bg-orange-50 rounded-lg">
                      <h3 className="font-bold text-lg text-orange-800 mb-2">{results.product_name || "Product Name Not Found"}</h3>
                      <p className="text-orange-700">Serving Size: {results.serving_size || "N/A"}</p>
                      <p className="text-2xl font-bold text-orange-600">{results.calories || 0} Calories</p>
                    </div>

                    {/* Nutrients */}
                    {results.nutrients && results.nutrients.length > 0 ? (
                      <div>
                        <h4 className="font-semibold mb-3 text-gray-800">Nutritional Facts</h4>
                        <div className="space-y-2">
                          {results.nutrients.map((nutrientItem, index) => ( // Iterate directly over the array
                            <div key={index} className="flex justify-between items-center p-3 bg-white rounded-lg border shadow-sm">
                              <span className="font-medium capitalize text-gray-700">{nutrientItem.nutrient_name}</span>
                              <div className="text-right">
                                <span className="font-semibold text-gray-800">{nutrientItem.amount}</span>
                                {nutrientItem.daily_value && nutrientItem.daily_value.trim() !== "" && (
                                  <span className="text-sm text-gray-500 ml-2">({nutrientItem.daily_value} DV)</span>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                        <div className="p-3 bg-white rounded-lg border text-sm text-gray-500">No specific nutritional facts found.</div>
                    )}


                    {/* Ingredients */}
                    {results.ingredients && results.ingredients.length > 0 && (
                      <div>
                        <h4 className="font-semibold mb-3 text-gray-800">Ingredients</h4>
                        <div className="p-3 bg-white rounded-lg border shadow-sm">
                          <p className="text-sm text-gray-700">{results.ingredients.join(", ")}</p>
                        </div>
                      </div>
                    )}

                    {/* Allergens */}
                    {results.allergens && results.allergens.length > 0 && (
                      <div>
                        <h4 className="font-semibold mb-3 text-gray-800">Allergens</h4>
                        <div className="flex flex-wrap gap-2">
                          {results.allergens.map((allergen, index) => (
                            <span key={index} className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm shadow-sm">
                              {allergen}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : !isLoading && ( // Only show this if not loading and no results
                  <div className="text-center text-gray-500 py-8">
                    <Zap className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                    <p>Upload a nutrition label image to extract information</p>
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