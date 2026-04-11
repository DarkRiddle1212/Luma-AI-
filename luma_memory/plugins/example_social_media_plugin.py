"""
Example plugin for social media memory entries.

This plugin demonstrates how to extend the Luma Memory Module with
custom validation, processing, and metadata for social media actions.

Usage:
    from luma_memory.plugins.plugin_loader import load_plugins_from_file
    from luma_memory.plugins.plugin_interface import get_global_registry
    
    # Load the plugin
    load_plugins_from_file("luma_memory/plugins/example_social_media_plugin.py")
    
    # The plugin is now registered and will handle social media actions
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, UTC
import re

from .plugin_interface import (
    MemoryEntryPlugin,
    PluginValidationError,
    PluginProcessingError
)
from ..models import MemoryEntry, SensitivityLevel


class SocialMediaPlugin(MemoryEntryPlugin):
    """
    Plugin for handling social media memory entries.
    
    This plugin provides custom validation and processing for social media
    actions like tweets, posts, and shares. It demonstrates:
    - Custom context validation
    - Automatic tag generation
    - Default sensitivity levels
    - Context enrichment
    """
    
    @property
    def name(self) -> str:
        return "social_media"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def supported_actions(self) -> List[str]:
        return ["tweet", "facebook_post", "instagram_post", "linkedin_post", "social_share"]
    
    def validate_context(self, context: Dict[str, Any], action: str) -> tuple[bool, Optional[str]]:
        """
        Validate social media context data.
        
        Required fields vary by action type:
        - tweet: content (max 280 chars), optional: media_urls, hashtags
        - facebook_post: content, optional: media_urls, visibility
        - instagram_post: media_urls (required), optional: caption, hashtags
        - linkedin_post: content, optional: media_urls
        - social_share: url (required), platform, optional: comment
        """
        if action == "tweet":
            # Validate tweet content
            if "content" not in context:
                return False, "Tweet must have 'content' field"
            
            content = context["content"]
            if not isinstance(content, str):
                return False, "Tweet content must be a string"
            
            if len(content) > 280:
                return False, f"Tweet content exceeds 280 characters (got {len(content)})"
            
            # Validate optional media URLs
            if "media_urls" in context:
                if not isinstance(context["media_urls"], list):
                    return False, "media_urls must be a list"
                if len(context["media_urls"]) > 4:
                    return False, "Tweet can have at most 4 media attachments"
        
        elif action == "instagram_post":
            # Instagram requires media
            if "media_urls" not in context:
                return False, "Instagram post must have 'media_urls' field"
            
            media_urls = context["media_urls"]
            if not isinstance(media_urls, list) or len(media_urls) == 0:
                return False, "Instagram post must have at least one media URL"
        
        elif action == "social_share":
            # Share requires URL
            if "url" not in context:
                return False, "Social share must have 'url' field"
            
            if "platform" not in context:
                return False, "Social share must specify 'platform'"
            
            # Validate URL format
            url = context["url"]
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                return False, "Invalid URL format"
        
        elif action in ["facebook_post", "linkedin_post"]:
            # These require content
            if "content" not in context:
                return False, f"{action} must have 'content' field"
            
            if not isinstance(context["content"], str):
                return False, f"{action} content must be a string"
        
        return True, None
    
    def process_before_storage(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Process social media entry before storage.
        
        Enriches the entry with:
        - Extracted hashtags
        - Extracted mentions
        - Character count
        - Media count
        - Engagement metadata placeholders
        """
        context = entry.context
        action = entry.action
        
        # Extract hashtags from content
        if "content" in context:
            content = context["content"]
            hashtags = self._extract_hashtags(content)
            if hashtags:
                context["extracted_hashtags"] = hashtags
            
            # Extract mentions
            mentions = self._extract_mentions(content)
            if mentions:
                context["extracted_mentions"] = mentions
            
            # Add character count
            context["character_count"] = len(content)
        
        # Add media count
        if "media_urls" in context:
            context["media_count"] = len(context["media_urls"])
        
        # Add engagement metadata placeholders
        context["engagement"] = {
            "likes": 0,
            "shares": 0,
            "comments": 0,
            "views": 0,
            "last_updated": datetime.now(UTC).isoformat()
        }
        
        # Add platform-specific metadata
        if action == "tweet":
            context["platform"] = "twitter"
            context["is_retweet"] = context.get("is_retweet", False)
            context["is_reply"] = context.get("is_reply", False)
        elif action == "facebook_post":
            context["platform"] = "facebook"
            context["visibility"] = context.get("visibility", "friends")
        elif action == "instagram_post":
            context["platform"] = "instagram"
        elif action == "linkedin_post":
            context["platform"] = "linkedin"
        elif action == "social_share":
            # Platform already specified in context
            pass
        
        return entry
    
    def get_default_sensitivity(self, action: str) -> Optional[SensitivityLevel]:
        """
        Get default sensitivity for social media actions.
        
        Social media posts are typically public, but we default to private
        to be conservative with user data.
        """
        # Default to private for safety
        return SensitivityLevel.PRIVATE
    
    def get_default_tags(self, action: str, context: Dict[str, Any]) -> List[str]:
        """
        Generate default tags for social media entries.
        
        Tags include:
        - Action type (tweet, post, etc.)
        - Platform name
        - Extracted hashtags
        - Media type (if applicable)
        """
        tags = ["social_media", action]
        
        # Add platform tag
        if "platform" in context:
            tags.append(f"platform:{context['platform']}")
        
        # Add hashtags as tags
        if "extracted_hashtags" in context:
            tags.extend(context["extracted_hashtags"])
        
        # Add media type tags
        if "media_urls" in context and context["media_urls"]:
            tags.append("has_media")
            
            # Try to determine media type from URLs
            for url in context["media_urls"]:
                if any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".gif"]):
                    tags.append("media:image")
                elif any(ext in url.lower() for ext in [".mp4", ".mov", ".avi"]):
                    tags.append("media:video")
        
        # Add engagement tag if this is a share or retweet
        if context.get("is_retweet") or context.get("is_reply") or action == "social_share":
            tags.append("engagement")
        
        return tags
    
    def should_summarize(self, entries: List[MemoryEntry]) -> bool:
        """
        Determine if social media entries should be summarized.
        
        Summarize if:
        - More than 20 entries from the same platform in the group
        - Entries span less than 24 hours (burst of activity)
        """
        if len(entries) < 20:
            return False
        
        # Check if entries are from the same platform
        platforms = set()
        for entry in entries:
            if "platform" in entry.context:
                platforms.add(entry.context["platform"])
        
        # Only summarize if all from same platform
        if len(platforms) != 1:
            return False
        
        # Check time span
        if entries:
            timestamps = [entry.timestamp for entry in entries]
            time_span = max(timestamps) - min(timestamps)
            
            # Summarize if within 24 hours
            if time_span.total_seconds() < 86400:  # 24 hours
                return True
        
        return False
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """
        Extract hashtags from text.
        
        Args:
            text: Text to extract hashtags from
        
        Returns:
            List of hashtags (without # symbol)
        """
        # Match hashtags: # followed by alphanumeric and underscores
        pattern = r'#(\w+)'
        matches = re.findall(pattern, text)
        return matches
    
    def _extract_mentions(self, text: str) -> List[str]:
        """
        Extract mentions from text.
        
        Args:
            text: Text to extract mentions from
        
        Returns:
            List of mentions (without @ symbol)
        """
        # Match mentions: @ followed by alphanumeric and underscores
        pattern = r'@(\w+)'
        matches = re.findall(pattern, text)
        return matches
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata with additional information."""
        metadata = super().get_metadata()
        metadata.update({
            "description": "Plugin for handling social media memory entries",
            "author": "Luma Memory Team",
            "features": [
                "Hashtag extraction",
                "Mention extraction",
                "Platform-specific validation",
                "Automatic tag generation",
                "Engagement tracking"
            ]
        })
        return metadata
