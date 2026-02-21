import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Array<{ source: string; content: string; relevance: number }>;
}

export default function Chat() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.content, stream: true }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessageContent = '';
      
      // Add empty assistant message
      setMessages(prev => [...prev, {
          id: 'temp-assistant', 
          role: 'assistant', 
          content: ''
      }]);

      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        const events = buffer.split('\n\n');
        // Keep the last part in buffer as it might be incomplete
        // If the chunk ends with \n\n, the last element is empty string, which is fine
        buffer = events.pop() || ''; 
        
        for (const eventBlock of events) {
            if (!eventBlock.trim()) continue;
            
            const lines = eventBlock.split('\n');
            let eventType = '';
            let data = '';
            
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.substring(7).trim();
                } else if (line.startsWith('data: ')) {
                    data = line.substring(6).trim();
                }
            }
            
            if (eventType === 'token') {
                try {
                    const parsed = JSON.parse(data);
                    assistantMessageContent += parsed.content;
                    
                    setMessages(prev => prev.map(msg => 
                        msg.id === 'temp-assistant' 
                            ? { ...msg, content: assistantMessageContent }
                            : msg
                    ));
                } catch (e) {
                    console.error("Error parsing token data", e);
                }
            } else if (eventType === 'done') {
                try {
                     const parsed = JSON.parse(data);
                     setMessages(prev => prev.map(msg => 
                        msg.id === 'temp-assistant'
                            ? { 
                                ...msg, 
                                citations: parsed.citations, 
                                id: parsed.session_id || Date.now().toString() 
                              }
                            : msg
                     ));
                } catch (e) {
                    console.error("Error parsing done data", e);
                }
            } else if (eventType === 'clarification') {
                 try {
                     const parsed = JSON.parse(data);
                     assistantMessageContent += `\n\n[Clarification Needed]: ${parsed.message}`;
                     setMessages(prev => prev.map(msg => 
                        msg.id === 'temp-assistant' 
                            ? { ...msg, content: assistantMessageContent }
                            : msg
                    ));
                } catch (e) {}
            }
        }
      }
      
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: 'Sorry, something went wrong. Please try again.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 bg-gray-50">
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 p-4 bg-white rounded-xl shadow-sm">
        {messages.length === 0 && (
            <div className="text-center text-gray-400 mt-20">
                <Bot size={48} className="mx-auto mb-4" />
                <p>Start a conversation with the AI Assistant</p>
            </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx(
              "flex gap-3 p-4 rounded-lg max-w-[85%]",
              message.role === 'user' 
                ? "bg-blue-600 text-white ml-auto" 
                : "bg-gray-100 text-gray-800 mr-auto"
            )}
          >
            <div className={clsx(
              "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
              message.role === 'user' ? "bg-blue-700" : "bg-white border border-gray-200"
            )}>
              {message.role === 'user' ? <User size={18} /> : <Bot size={18} className="text-blue-600" />}
            </div>
            <div className="flex-1 min-w-0">
              <div className="prose prose-sm max-w-none break-words whitespace-pre-wrap">
                {message.content}
              </div>
              {message.citations && message.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200/50">
                  <p className="text-xs font-semibold opacity-70 mb-2">Sources:</p>
                  <div className="space-y-2">
                    {message.citations.map((cite, idx) => (
                      <div key={idx} className="text-xs bg-black/5 p-2 rounded">
                        <span className="font-bold block mb-0.5">{cite.source}</span>
                        <span className="opacity-80 line-clamp-2">{cite.content}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        {isLoading && messages[messages.length - 1]?.role === 'user' && (
           <div className="flex justify-start ml-12">
             <Loader2 className="animate-spin text-blue-600" />
           </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 bg-white p-2 rounded-xl shadow-sm border border-gray-200">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the codebase..."
          className="flex-1 p-3 bg-transparent focus:outline-none"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Send size={20} />
        </button>
      </form>
    </div>
  );
}
