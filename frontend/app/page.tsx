"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Camera,
  Eye,
  Hash,
  Leaf,
  Zap,
  Calendar,
  ArrowRight,
  Sparkles,
  Shield,
  CloudLightningIcon as Lightning,
  TrendingUp,
} from "lucide-react"
import Link from "next/link"
import { useState } from "react"

export default function HomePage() {
  const [hoveredCard, setHoveredCard] = useState<string | null>(null)

  const features = [
    {
      id: "brand-recognition",
      title: "Brand Recognition",
      description: "Identify and classify product brands using advanced computer vision and deep learning algorithms",
      icon: <Eye className="h-8 w-8" />,
      color: "bg-gradient-to-br from-blue-500 to-blue-600",
      href: "/brand-recognition",
      stats: "99.2% Accuracy",
      category: "Computer Vision",
      gradient: "from-blue-50 to-blue-100",
    },
    {
      id: "freshness-detection",
      title: "Freshness Detection",
      description: "Analyze fruits and vegetables to determine freshness levels using AI-powered quality assessment",
      icon: <Leaf className="h-8 w-8" />,
      color: "bg-gradient-to-br from-green-500 to-green-600",
      href: "/freshness-detection",
      stats: "Real-time Analysis",
      category: "Quality Control",
      gradient: "from-green-50 to-green-100",
    },
    {
      id: "item-counting",
      title: "Count Items",
      description: "Automatically count the number of items in warehouse images with precision object detection",
      icon: <Hash className="h-8 w-8" />,
      color: "bg-gradient-to-br from-purple-500 to-purple-600",
      href: "/item-counting",
      stats: "Instant Counting",
      category: "Inventory",
      gradient: "from-purple-50 to-purple-100",
    },
    {
      id: "nutrition-info",
      title: "Nutrition Info",
      description: "Extract comprehensive nutritional information from food product labels using advanced OCR",
      icon: <Zap className="h-8 w-8" />,
      color: "bg-gradient-to-br from-orange-500 to-orange-600",
      href: "/nutrition-info",
      stats: "Multi-language",
      category: "Health & Safety",
      gradient: "from-orange-50 to-orange-100",
    },
    {
      id: "mrp-expiry",
      title: "MRP & Expiry Detection",
      description: "Extract MRP, expiry dates, and manufacturing details from product labels with high precision",
      icon: <Calendar className="h-8 w-8" />,
      color: "bg-gradient-to-br from-red-500 to-red-600",
      href: "/mrp-expiry",
      stats: "Smart Alerts",
      category: "Compliance",
      gradient: "from-red-50 to-red-100",
    },
  ]

  const stats = [
    { label: "Products Analyzed", value: "50K+", icon: <TrendingUp className="h-5 w-5" /> },
    { label: "Accuracy Rate", value: "99.2%", icon: <Shield className="h-5 w-5" /> },
    { label: "Processing Speed", value: "<2s", icon: <Lightning className="h-5 w-5" /> },
    { label: "AI Models", value: "5+", icon: <Sparkles className="h-5 w-5" /> },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 bg-grid-slate-100 [mask-image:linear-gradient(0deg,white,rgba(255,255,255,0.6))] -z-10" />

        <div className="container mx-auto px-4 py-16">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="flex items-center justify-center mb-6">
              <div className="relative">
                <div className="absolute inset-0 bg-blue-600 rounded-full blur-xl opacity-20 animate-pulse" />
                <div className="relative bg-gradient-to-br from-blue-500 to-blue-600 p-4 rounded-2xl shadow-lg">
                  <Camera className="h-12 w-12 text-white" />
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <Badge
                variant="secondary"
                className="px-4 py-2 text-sm font-medium bg-blue-50 text-blue-700 border-blue-200"
              >
                <Sparkles className="h-4 w-4 mr-2" />
                Powered by Advanced AI
              </Badge>

              <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 bg-clip-text text-transparent leading-tight">
                Warehouse Vision
              </h1>

              <p className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                Transform your warehouse operations with cutting-edge computer vision and machine learning technology
              </p>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-12 max-w-4xl mx-auto">
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <div className="flex items-center justify-center mb-2 text-blue-600">{stat.icon}</div>
                  <div className="text-2xl font-bold text-gray-900">{stat.value}</div>
                  <div className="text-sm text-gray-600">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Features Grid */}
          <div className="mb-16">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">Powerful AI Features</h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Choose from our suite of advanced computer vision tools designed for modern warehouse management
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
              {features.map((feature, index) => (
                <Card
                  key={feature.id}
                  className={`group relative overflow-hidden border-0 shadow-lg hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-2 bg-gradient-to-br ${feature.gradient}`}
                  onMouseEnter={() => setHoveredCard(feature.id)}
                  onMouseLeave={() => setHoveredCard(null)}
                >
                  {/* Animated background */}
                  <div className="absolute inset-0 bg-gradient-to-br from-white/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                  {/* Floating elements */}
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-all duration-500 transform translate-x-4 group-hover:translate-x-0">
                    <ArrowRight className="h-5 w-5 text-gray-400" />
                  </div>

                  <CardHeader className="relative pb-4">
                    <div className="flex items-start justify-between mb-4">
                      <div
                        className={`p-3 rounded-xl ${feature.color} shadow-lg group-hover:scale-110 transition-transform duration-300`}
                      >
                        <div className="text-white">{feature.icon}</div>
                      </div>
                      <Badge variant="outline" className="text-xs font-medium bg-white/80 backdrop-blur-sm">
                        {feature.category}
                      </Badge>
                    </div>

                    <CardTitle className="text-xl font-bold text-gray-900 group-hover:text-gray-800 transition-colors">
                      {feature.title}
                    </CardTitle>

                    <CardDescription className="text-gray-600 leading-relaxed">{feature.description}</CardDescription>
                  </CardHeader>

                  <CardContent className="relative">
                    <div className="flex items-center justify-between mb-4">
                      <Badge variant="secondary" className="bg-white/60 text-gray-700 font-medium">
                        {feature.stats}
                      </Badge>
                    </div>

                    <Link href={feature.href}>
                      <Button
                        className="w-full group-hover:bg-gray-900 transition-all duration-300 shadow-md hover:shadow-lg"
                        size="lg"
                      >
                        <span className="mr-2">Get Started</span>
                        <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
                      </Button>
                    </Link>
                  </CardContent>

                  {/* Hover effect overlay */}
                  <div
                    className={`absolute inset-0 bg-gradient-to-br from-transparent to-black/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none`}
                  />
                </Card>
              ))}
            </div>
          </div>

          {/* CTA Section */}
          <div className="text-center bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl p-12 text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-black/10 backdrop-blur-sm" />
            <div className="relative z-10">
              <h3 className="text-3xl font-bold mb-4">Ready to Transform Your Warehouse?</h3>
              <p className="text-xl mb-8 text-blue-100 max-w-2xl mx-auto">
                Join thousands of businesses using AI-powered warehouse management solutions
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button size="lg" variant="secondary" className="bg-white text-blue-600 hover:bg-gray-100">
                  <Camera className="h-5 w-5 mr-2" />
                  Start Free Trial
                </Button>
                <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10">
                  View Documentation
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center mb-6">
            <Camera className="h-8 w-8 text-blue-400 mr-3" />
            <span className="text-2xl font-bold">Warehouse Vision</span>
          </div>
          <p className="text-gray-400 mb-6">
            Powered by advanced machine learning models and computer vision technology
          </p>
          <div className="flex justify-center space-x-8 text-sm text-gray-400">
            <span>© 2024 Warehouse Vision</span>
            <span>•</span>
            <span>Privacy Policy</span>
            <span>•</span>
            <span>Terms of Service</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
