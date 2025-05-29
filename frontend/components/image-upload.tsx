"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button" // Assuming this path is correct
import { Card, CardContent } from "@/components/ui/card" // Assuming this path is correct
import { Upload, Camera, X, Loader2, AlertCircle, Sparkles, Tags, FileText } from "lucide-react"

interface ImageUploadProps {
  onImageSelect: (file: File) => void
  onAnalyze: (file: File) => Promise<void> 
  isLoading?: boolean // This is for the onAnalyze prop, not camera starting
}

export default function ImageUpload({ onImageSelect, onAnalyze, isLoading = false }: ImageUploadProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [showCamera, setShowCamera] = useState(false)
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isStartingCamera, setIsStartingCamera] = useState(false)

  // State for Gemini API features (logic kept, UI temporarily adjusted)
  const [imageDescription, setImageDescription] = useState<string | null>(null);
  const [isDescribing, setIsDescribing] = useState(false);
  const [descriptionError, setDescriptionError] = useState<string | null>(null);

  const [suggestedTags, setSuggestedTags] = useState<string[] | null>(null);
  const [isSuggestingTags, setIsSuggestingTags] = useState(false);
  const [tagsError, setTagsError] = useState<string | null>(null);
  const [base64Image, setBase64Image] = useState<string | null>(null);


  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Helper function to convert File to Base64
  const convertFileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64String = (reader.result as string).split(',')[1];
        resolve(base64String);
      };
      reader.onerror = (error) => reject(error);
    });
  };
  
  useEffect(() => {
    if (selectedImage) {
      console.log("Selected image changed, converting to base64...");
      convertFileToBase64(selectedImage)
        .then(b64 => { setBase64Image(b64); console.log("Base64 conversion successful."); })
        .catch(err => { console.error("Error converting image to base64:", err); setCameraError("Could not process image for analysis."); setBase64Image(null); });
    } else { setBase64Image(null); }
  }, [selectedImage]);


  useEffect(() => {
    const videoElement = videoRef.current;
    // This effect runs when cameraStream or showCamera changes.
    // videoRef.current should be available if showCamera is true (due to JSX structure).
    if (videoElement && cameraStream && showCamera) {
      console.log("useEffect [cameraStream, showCamera]: Attaching stream to video element. Video Ref:", videoElement);
      videoElement.srcObject = cameraStream;

      const handleCanPlay = () => {
        console.log("Video event: 'canplay'. Attempting to play.");
        videoElement.play()
          .then(() => { console.log("Video event: play() promise resolved. Waiting for 'playing' event.");})
          .catch(handlePlayPromiseError);
      };
      
      const handlePlaying = () => {
        console.log("Video event: 'playing' - playback has truly begun.");
        setIsStartingCamera(false); 
      };

      const handlePlayPromiseError = (error: any) => {
        console.error("useEffect: video.play() promise rejected:", error.name, error.message);
        let specificErrorMessage = "Failed to play video. Ensure permissions and no other app is using camera.";
        if (error.name === "NotAllowedError") specificErrorMessage = "Video playback prevented. Try clicking 'Use Camera' again.";
        else if (error.name === "NotFoundError") specificErrorMessage = "No video source found for playback.";
        else if (error.name === "AbortError") specificErrorMessage = "Video playback aborted.";
        setCameraError(specificErrorMessage);
        setIsStartingCamera(false);
      };

      const handleVideoElementGenericError = (event: Event) => {
        console.error("useEffect: Video element 'error' event:", event);
        const videoError = (event.target as HTMLVideoElement).error;
        let errorMsg = "Video display error.";
        if (videoError) errorMsg += ` Code: ${videoError.code}, Message: ${videoError.message}`;
        setCameraError(errorMsg);
        setIsStartingCamera(false);
      };
      
      videoElement.addEventListener('canplay', handleCanPlay);
      videoElement.addEventListener('playing', handlePlaying);
      videoElement.addEventListener('error', handleVideoElementGenericError);
      
      // Manually calling load() can sometimes help ensure the new srcObject is processed.
      videoElement.load(); 
      console.log("useEffect: Called videoElement.load()");


      return () => {
        console.log("useEffect cleanup [cameraStream, showCamera]: Removing event listeners for stream:", cameraStream?.id);
        videoElement.removeEventListener('canplay', handleCanPlay);
        videoElement.removeEventListener('playing', handlePlaying);
        videoElement.removeEventListener('error', handleVideoElementGenericError);
      };
    } else {
        console.log("useEffect [cameraStream, showCamera]: Did not run main logic. Conditions:", 
            { hasVideoElement: !!videoElement, hasCameraStream: !!cameraStream, showCamera });
    }
  }, [cameraStream, showCamera]);


  const processImageSelection = async (file: File) => {
    console.log("Processing image selection:", file.name);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedImage(file);
    const newPreviewUrl = URL.createObjectURL(file);
    setPreviewUrl(newPreviewUrl);
    onImageSelect(file); 
    setImageDescription(null); setDescriptionError(null);
    setSuggestedTags(null); setTagsError(null);
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) processImageSelection(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) processImageSelection(file);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => event.preventDefault();

  const startCamera = async () => {
    console.log("startCamera called.");
    setIsStartingCamera(true);
    setCameraError(null); 
    
    if (cameraStream) { 
      console.log("Stopping existing camera stream before starting new one.");
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null); 
    }
    // Do not clear videoRef.current.srcObject here if the video tag is always mounted when showCamera=true
    // The useEffect will handle new stream.

    setShowCamera(true); // This will ensure the <video> tag is in the DOM for the ref.

    try {
      console.log("Requesting camera access via getUserMedia...");
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("getUserMedia not supported.");
      }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      console.log("Camera stream received:", stream.id);
      setCameraStream(stream); 
    } catch (error: any) {
      console.error("Camera access error (getUserMedia):", error.name, error.message);
      let msg = "Camera access failed. Check permissions & ensure camera is not in use.";
      if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") msg = "Permission denied. Allow camera access in browser settings and refresh.";
      else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") msg = "No camera found. Ensure a camera is connected and enabled.";
      else if (error.name === "NotSupportedError" || error.name === "SourceUnavailableError") msg = "Camera not supported or currently unavailable.";
      else if (error.name === "AbortError") msg = "Camera access aborted.";
      else if (error.name === "OverconstrainedError") msg = `Camera doesn't support requested settings. Error: ${error.message}`;
      else if (error.name === "TypeError") msg = "Camera access failed (TypeError). Ensure HTTPS.";
      
      setCameraError(msg);
      setIsStartingCamera(false); 
      setShowCamera(false); 
    }
  };

  const capturePhoto = () => { /* ... (no changes to this function's core logic) ... */ 
    console.log("capturePhoto called.");
    if (!videoRef.current || !canvasRef.current || !videoRef.current.srcObject || videoRef.current.readyState < videoRef.current.HAVE_METADATA) {
      console.error("Camera not ready for capture. VideoRef:", videoRef.current, "CanvasRef:", canvasRef.current);
      setCameraError("Could not capture: Camera not ready or video stream not active.");
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const videoWidth = video.videoWidth, videoHeight = video.videoHeight;

    if (videoWidth === 0 || videoHeight === 0) {
      console.warn("Video dimensions zero at capture. Photo might be blank. Video state:", video.readyState);
      setCameraError("Video has no valid dimensions. Cannot capture photo.");
      return;
    }
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    console.log("Canvas dimensions for capture set to:", canvas.width, "x", canvas.height);
    const ctx = canvas.getContext("2d");

    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      console.log("Image drawn to canvas.");
      canvas.toBlob(blob => {
        if (blob) {
          const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
          processImageSelection(file); 
          console.log("Photo captured successfully. Stopping camera.");
          stopCamera();
        } else {
          console.error("Failed to create blob from canvas.");
          setCameraError("Could not process captured photo (blob creation failed).");
        }
      }, "image/jpeg", 0.9);
    } else {
      console.error("Failed to get 2D context from canvas.");
      setCameraError("Could not prepare image capture (canvas context error).");
    }
  }


  const stopCamera = () => { /* ... (no changes to this function's core logic) ... */ 
    console.log("stopCamera called.");
    if (cameraStream) {
      console.log("Stopping all tracks for stream:", cameraStream.id);
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null); 
    }
    if (videoRef.current) {
      console.log("Clearing video srcObject, pausing, and loading null.");
      videoRef.current.srcObject = null; 
      videoRef.current.pause(); 
      if (videoRef.current.src !== "") videoRef.current.src = ""; 
      videoRef.current.load(); 
    }
    setShowCamera(false); 
    setIsStartingCamera(false); 
  }

  const clearImage = () => { /* ... (no changes to this function's core logic) ... */ 
    console.log("clearImage called.");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedImage(null);
    setPreviewUrl(null);
    setBase64Image(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (showCamera || cameraStream) {
        console.log("Clearing image, also stopping active camera.");
        stopCamera();
    }
    setCameraError(null); 
    setImageDescription(null); setDescriptionError(null);
    setSuggestedTags(null); setTagsError(null);
  }

  const handleDescribeImage = async () => { /* ... (Gemini logic, no UI changes here) ... */ };
  const handleSuggestTags = async () => { /* ... (Gemini logic, no UI changes here) ... */ };

  useEffect(() => {
    return () => {
      console.log("ImageUpload component unmounting. Cleaning up.");
      stopCamera(); 
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewUrl]); 

  return (
    <div className="space-y-6 p-4 md:p-6 bg-gray-100 min-h-screen">
      <canvas ref={canvasRef} className="hidden" />

      {/* Initial Upload State */}
      {!selectedImage && !showCamera && (
        <Card className="shadow-xl rounded-xl bg-white">
          <CardContent className="p-6 md:p-8">
            <div
              className="border-2 border-dashed border-gray-300 hover:border-blue-500 rounded-lg p-6 md:p-10 text-center transition-colors duration-200 ease-in-out cursor-pointer bg-gray-50 hover:bg-gray-100"
              onDrop={handleDrop} onDragOver={handleDragOver} onClick={() => !cameraError && fileInputRef.current?.click()}
            >
              <Upload className="h-12 w-12 md:h-16 md:w-16 text-gray-400 mx-auto mb-4" />
              <p className="text-lg md:text-xl font-semibold text-gray-700 mb-2">Upload an image or take a photo</p>
              <p className="text-sm text-gray-500 mb-6">Drag & drop, click to browse, or use your camera.</p>
              {cameraError && !isStartingCamera && ( // Show general camera errors here if not in starting phase
                <div className="my-4 p-3 bg-red-50 border border-red-300 rounded-lg text-red-700 text-sm">
                  <div className="flex items-center"><AlertCircle className="h-5 w-5 mr-2 shrink-0" /><span>{cameraError}</span></div>
                  <Button onClick={(e) => { e.stopPropagation(); setCameraError(null); }} size="sm" variant="outline" className="mt-2 border-red-300 text-red-600 hover:bg-red-100">Dismiss</Button>
                </div>
              )}
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Button onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }} variant="outline" className="w-full sm:w-auto"><Upload className="h-4 w-4 mr-2" />Choose File</Button>
                <Button onClick={(e) => { e.stopPropagation(); startCamera(); }} variant="outline" className="w-full sm:w-auto"><Camera className="h-4 w-4 mr-2" />Use Camera</Button>
              </div>
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Camera View (includes starting state and active state) */}
      {showCamera && (
        <Card className="shadow-xl rounded-xl overflow-hidden">
          <CardContent className="p-0">
            <div className="text-center space-y-0">
              {/* Video display area */}
              <div className="relative w-full bg-black aspect-video md:max-h-[calc(100vh-200px)]"> {/* Adjusted max height */}
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover" // Video tag is always rendered if showCamera is true
                />
                {/* Loading Overlay */}
                {isStartingCamera && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-800 bg-opacity-85 text-white p-4">
                    <Loader2 className="h-10 w-10 text-blue-400 animate-spin mb-4" />
                    <p className="text-lg font-medium">Starting camera...</p>
                    <p className="text-sm text-gray-300">Please allow camera access.</p>
                    {cameraError && ( // Show specific startup error if any
                        <div className="mt-3 p-2 bg-red-200 text-red-800 text-xs rounded max-w-xs text-center">
                            {cameraError}
                        </div>
                    )}
                  </div>
                )}
                {/* Error Overlay for playback issues after starting attempt */}
                {!isStartingCamera && cameraError && ( // Show error if not starting but an error exists
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-85 p-4">
                    <AlertCircle className="h-8 w-8 text-red-400 mb-2" />
                    <p className="text-red-400 text-center text-sm">{cameraError}</p>
                    <Button onClick={() => { setCameraError(null); startCamera();}} variant="outline" size="sm" className="mt-3 bg-white text-gray-700 hover:bg-gray-100">Try Again</Button>
                  </div>
                )}
              </div>

              {/* Controls Area */}
              <div className="p-4 md:p-6 bg-gray-800 rounded-b-lg border-t border-gray-700">
                {isStartingCamera ? ( 
                    <Button onClick={stopCamera} variant="outline" className="text-gray-300 border-gray-600 hover:bg-gray-700 hover:text-white w-full sm:w-auto">Cancel</Button>
                ) : cameraStream && !cameraError ? ( // Show Capture only if stream is active, not starting, and no error
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                        <Button onClick={capturePhoto} size="lg" className="bg-blue-600 hover:bg-blue-700 text-white w-full sm:w-auto"><Camera className="h-5 w-5 mr-2" />Capture Photo</Button>
                        <Button onClick={stopCamera} variant="outline" className="text-gray-300 border-gray-600 hover:bg-gray-700 hover:text-white w-full sm:w-auto">Cancel</Button>
                    </div>
                ) : ( // Fallback if no stream or error but not starting (e.g. after a failed start)
                    <Button onClick={stopCamera} variant="outline" className="text-gray-300 border-gray-600 hover:bg-gray-700 hover:text-white w-full sm:w-auto">Close Camera</Button>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Image Preview State (after capture/upload, camera closed) */}
      {selectedImage && previewUrl && !showCamera && (
        <Card className="shadow-xl rounded-xl bg-white">
          <CardContent className="p-6 md:p-8">
            <div className="text-center">
              <p className="text-2xl font-semibold text-gray-800 mb-6">Image Preview & Analysis</p>
              <div className="relative inline-block mb-6 group w-full max-w-lg mx-auto">
                <img src={previewUrl} alt="Selected preview" width={400} height={300}
                  className="rounded-lg object-contain w-full max-h-96 border border-gray-300 shadow-md bg-gray-50"
                  onError={(e) => { (e.target as HTMLImageElement).src = "https://placehold.co/400x300/e0e0e0/757575?text=Preview+Error"; }}
                  style={{ aspectRatio: '4/3' }} />
                <Button onClick={clearImage} size="icon" variant="destructive" className="absolute top-3 right-3 rounded-full opacity-80 group-hover:opacity-100 transition-opacity" aria-label="Clear image"><X className="h-5 w-5" /></Button>
              </div>
              
              <div className="mb-6">
                <Button onClick={() => selectedImage && onAnalyze(selectedImage)} disabled={isLoading || !selectedImage} size="lg" className="w-full sm:w-auto">
                  {isLoading ? <><Loader2 className="h-5 w-5 mr-2 animate-spin" />Analyzing for Brands...</> : "Analyze for Brands"}
                </Button>
              </div>
              {/* Gemini UI (still commented out for camera focus) */}
              <Button onClick={clearImage} variant="outline" className="mt-8 w-full sm:w-auto">
                Clear and Upload New Image
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
