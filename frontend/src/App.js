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

        if (!emailContent.length < 50) {
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

}