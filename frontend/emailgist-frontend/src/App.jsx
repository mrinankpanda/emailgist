import React, {useState} from "react";
import {Mail, Sparkles, Clock, AlertCircle, CheckCircle} from "lucide-react";

const EmailGist = () => {
    const [emailContent, setEmailContent] = useState('');
    const [summary, setSummary] = useState('');
    const [highlights, setHighlights] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [processingTime, setProcessingTime] = useState(null);
    const [error, setError] = useState('');

    const API_URL = 'http://localhost:8000';

    const handleSubmit = async () => {
        if (!emailContent.trim()) {
            setError('Please enter some email content to summarize.');
            return;
        }

        if (emailContent.length < 50) {
            setError('Email content must be at least 50 characters long.');
            return;
        }

        setIsLoading(true);
        setError('');
        setSummary('');
        setHighlights([]);
        setProcessingTime(null);

        try {
            const response = await fetch(`${API_URL}/summarize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({email_content: emailContent}),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to summarize email.');
            }

            const data = await response.json();
            setSummary(data.summary);
            setHighlights(data.highlights);
            setProcessingTime(data.processing_time);
        } catch (err) {
            setError(err.message || 'Failed to connect to the server. Make sure that the backend is running on port 8000.');
        } finally {
            setIsLoading(false);
        }
    };

    const clearAll = () => {
        setEmailContent('');
        setSummary('');
        setHighlights([]);
        setProcessingTime(null);
        setError('');
    };

    const handleKeyDown = (e) => {
        if (e.key == 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            handleSubmit();
        }
    };

return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Mail className="h-10 w-10 text-indigo-600" />
            <h1 className="text-4xl font-bold text-gray-900">EmailGist</h1>
            <Sparkles className="h-8 w-8 text-yellow-500" />
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Transform lengthy emails into actionable insights using AI-powered summarization
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="space-y-6">
            {/* Email Input */}
            <div>
              <label htmlFor="email-content" className="block text-sm font-semibold text-gray-700 mb-2">
                📧 Paste Your Email Content
              </label>
              <textarea
                id="email-content"
                value={emailContent}
                onChange={(e) => setEmailContent(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Paste your lengthy email here... (minimum 10 characters)&#10;&#10;Tip: Use Ctrl+Enter to submit quickly!"
                rows={8}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-colors resize-none"
              />
              <div className="text-right text-sm text-gray-500 mt-1">
                {emailContent.length} characters
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={handleSubmit}
                disabled={isLoading || !emailContent.trim()}
                className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-5 w-5" />
                    Summarize Email
                  </>
                )}
              </button>
              
              <button
                onClick={clearAll}
                className="px-6 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors"
              >
                Clear All
              </button>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border-l-4 border-red-400 rounded-r-lg">
              <div className="flex items-center">
                <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          )}

          {/* Results Section */}
          {(summary || highlights.length > 0) && (
            <div className="mt-8 space-y-6">
              {/* Processing Time */}
              {processingTime && (
                <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
                  <Clock className="h-4 w-4" />
                  <span>Processed in {processingTime}s</span>
                  <CheckCircle className="h-4 w-4 text-green-500" />
                </div>
              )}

              {/* Summary */}
              {summary && (
                <div className="bg-gradient-to-r from-indigo-50 to-blue-50 p-6 rounded-xl border-l-4 border-indigo-400">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-indigo-600" />
                    AI Summary
                  </h3>
                  <p className="text-gray-800 leading-relaxed">{summary}</p>
                </div>
              )}

              {/* Highlights */}
              {highlights.length > 0 && (
                <div className="bg-yellow-50 p-6 rounded-xl border-l-4 border-yellow-400">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">
                    🎯 Key Highlights
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {highlights.map((highlight, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-yellow-200 text-yellow-800 rounded-full text-sm font-medium"
                      >
                        {highlight}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500">
          <p>Made with ❤️ for productivity enthusiasts</p>
          <p className="text-sm mt-1">Backend Status: {isLoading ? 'Processing...' : 'Ready'}</p>
        </div>
      </div>
    </div>
  );
};

export default EmailGist;