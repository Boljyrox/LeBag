import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plane, Luggage, User } from 'lucide-react';

const API_URL = "http://localhost:5001/api/bags";

function App() {
  const [bags, setBags] = useState([]);

  useEffect(() => {
    const fetchBags = async () => {
      try {
        const response = await fetch(API_URL);
        if (response.ok) {
          const data = await response.json();
          // Assuming data is sorted newest first by backend
          setBags(data);
        }
      } catch (error) {
        console.error("Error fetching bags:", error);
      }
    };

    // Initial fetch
    fetchBags();

    // Poll every 1 second
    const interval = setInterval(fetchBags, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen p-8 flex flex-col items-center relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-500/20 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-blue-500/20 rounded-full blur-[100px]" />
      </div>

      {/* Header */}
      <header className="mb-12 text-center">
        <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent drop-shadow-sm">
          Bags Ready For Collection
        </h1>
        <p className="mt-4 text-gray-400 text-lg">Real-time Luggage Tracking System</p>
      </header>

      {/* List */}
      <div className="w-full max-w-2xl space-y-4">
        <AnimatePresence mode='popLayout'>
          {bags.map((bag) => (
            <motion.div
              key={bag.id}
              layout
              initial={{ opacity: 0, y: -50, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.2 } }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="glass p-6 rounded-2xl flex items-center justify-between group hover:bg-white/15 transition-colors duration-300"
            >
              {/* Left: Icon & Owner */}
              <div className="flex items-center space-x-4">
                <div className="p-3 bg-white/10 rounded-full text-white/80 group-hover:text-white group-hover:scale-110 transition-all duration-300">
                  <Luggage size={24} />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white group-hover:text-blue-200 transition-colors">
                    {bag.owner}
                  </h2>
                  <p className="text-sm text-gray-400 flex items-center gap-1">
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse inline-block mr-1"></span>
                    {bag.type}
                  </p>
                </div>
              </div>

              {/* Right: Flight Info */}
              <div className="flex flex-col items-end">
                <div className="flex items-center space-x-2 bg-black/40 px-3 py-1 rounded-full border border-white/10 shadow-inner">
                  <Plane size={14} className="text-blue-400 transform -rotate-45" />
                  <span className="text-sm font-mono font-semibold tracking-wider text-blue-100">
                    {bag.flight}
                  </span>
                </div>
                <span className="text-xs text-gray-500 mt-2 font-mono">
                  {new Date(bag.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {bags.length === 0 && (
          <motion.div 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }}
            className="text-center text-gray-500 py-12"
          >
            No bags currently on the belt...
          </motion.div>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-auto pt-12 pb-4 text-center">
        <p className="text-gray-600 font-medium tracking-widest text-sm uppercase">LeBag System v1.0</p>
      </footer>
    </div>
  );
}

export default App;
