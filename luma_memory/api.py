"""
REST API for memory module - enables agent communication.
"""

from flask import Flask, request, jsonify
from datetime import datetime
from typing import Optional
import logging

from .memory_manager import MemoryManager
from .models import MemoryType


class MemoryAPI:
    """REST API wrapper for the memory manager."""
    
    def __init__(self, memory_manager: MemoryManager, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize the API.
        
        Args:
            memory_manager: MemoryManager instance
            host: Host to bind to
            port: Port to bind to
        """
        self.app = Flask(__name__)
        self.memory_manager = memory_manager
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API endpoints."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({"status": "healthy", "service": "luma-memory"}), 200
        
        @self.app.route('/memory', methods=['POST'])
        def store_memory():
            """
            Store a new memory entry.
            
            Expected JSON body:
            {
                "content": "string",
                "memory_type": "action|context|conversation|task|system",
                "source": "string",
                "metadata": {},  # optional
                "tags": [],  # optional
                "encrypt": false  # optional
            }
            """
            try:
                data = request.get_json()
                
                # Validate required fields
                if not data or 'content' not in data or 'memory_type' not in data or 'source' not in data:
                    return jsonify({
                        "error": "Missing required fields: content, memory_type, source"
                    }), 400
                
                # Parse memory type
                try:
                    memory_type = MemoryType(data['memory_type'])
                except ValueError:
                    return jsonify({
                        "error": f"Invalid memory_type. Must be one of: {[t.value for t in MemoryType]}"
                    }), 400
                
                # Store memory
                entry_id = self.memory_manager.store_memory(
                    content=data['content'],
                    memory_type=memory_type,
                    source=data['source'],
                    metadata=data.get('metadata'),
                    tags=data.get('tags'),
                    encrypt=data.get('encrypt', False)
                )
                
                if entry_id:
                    return jsonify({
                        "success": True,
                        "entry_id": entry_id
                    }), 201
                else:
                    return jsonify({
                        "error": "Failed to store memory"
                    }), 500
                    
            except Exception as e:
                self.logger.error(f"Error in store_memory: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/memory', methods=['GET'])
        def retrieve_memories():
            """
            Retrieve memory entries with optional filters.
            
            Query parameters:
            - start_time: ISO format datetime
            - end_time: ISO format datetime
            - memory_type: action|context|conversation|task|system
            - source: string
            - tags: comma-separated list
            - limit: integer (default 100)
            """
            try:
                # Parse query parameters
                start_time = None
                end_time = None
                memory_type = None
                source = request.args.get('source')
                tags = None
                limit = int(request.args.get('limit', 100))
                
                if request.args.get('start_time'):
                    start_time = datetime.fromisoformat(request.args.get('start_time'))
                
                if request.args.get('end_time'):
                    end_time = datetime.fromisoformat(request.args.get('end_time'))
                
                if request.args.get('memory_type'):
                    try:
                        memory_type = MemoryType(request.args.get('memory_type'))
                    except ValueError:
                        return jsonify({
                            "error": f"Invalid memory_type. Must be one of: {[t.value for t in MemoryType]}"
                        }), 400
                
                if request.args.get('tags'):
                    tags = request.args.get('tags').split(',')
                
                # Retrieve memories
                entries = self.memory_manager.retrieve_memories(
                    start_time=start_time,
                    end_time=end_time,
                    memory_type=memory_type,
                    source=source,
                    tags=tags,
                    limit=limit
                )
                
                # Convert to JSON-serializable format
                result = [entry.to_dict() for entry in entries]
                
                return jsonify({
                    "success": True,
                    "count": len(result),
                    "entries": result
                }), 200
                
            except Exception as e:
                self.logger.error(f"Error in retrieve_memories: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/memory/<entry_id>', methods=['GET'])
        def get_memory(entry_id: str):
            """Retrieve a specific memory entry by ID."""
            try:
                entry = self.memory_manager.get_memory(entry_id)
                
                if entry:
                    return jsonify({
                        "success": True,
                        "entry": entry.to_dict()
                    }), 200
                else:
                    return jsonify({
                        "error": "Memory entry not found"
                    }), 404
                    
            except Exception as e:
                self.logger.error(f"Error in get_memory: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/memory/<entry_id>', methods=['DELETE'])
        def delete_memory(entry_id: str):
            """Delete a memory entry."""
            try:
                success = self.memory_manager.delete_memory(entry_id)
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": "Memory entry deleted"
                    }), 200
                else:
                    return jsonify({
                        "error": "Memory entry not found"
                    }), 404
                    
            except Exception as e:
                self.logger.error(f"Error in delete_memory: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/memory/summary', methods=['GET'])
        def get_summary():
            """
            Get a summary of memory context.
            
            Query parameters:
            - start_time: ISO format datetime
            - end_time: ISO format datetime
            - source: string
            """
            try:
                start_time = None
                end_time = None
                source = request.args.get('source')
                
                if request.args.get('start_time'):
                    start_time = datetime.fromisoformat(request.args.get('start_time'))
                
                if request.args.get('end_time'):
                    end_time = datetime.fromisoformat(request.args.get('end_time'))
                
                summary = self.memory_manager.summarize_context(
                    start_time=start_time,
                    end_time=end_time,
                    source=source
                )
                
                return jsonify({
                    "success": True,
                    "summary": summary
                }), 200
                
            except Exception as e:
                self.logger.error(f"Error in get_summary: {e}")
                return jsonify({"error": str(e)}), 500
    
    def run(self, debug: bool = False):
        """
        Start the API server.
        
        Args:
            debug: Enable debug mode
        """
        self.logger.info(f"Starting Memory API on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
