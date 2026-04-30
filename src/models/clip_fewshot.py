"""CLIP-based few-shot learning model."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from ..utils.core import get_device

logger = logging.getLogger(__name__)


class CLIPFewShotLearner:
    """CLIP-based few-shot learning model.
    
    This class implements few-shot learning using pre-trained CLIP models.
    It supports both zero-shot and few-shot classification.
    
    Args:
        model_name: Name of the CLIP model to use.
        device: Device to run the model on.
        precision: Model precision (fp16, fp32).
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        precision: str = "fp32",
    ):
        self.model_name = model_name
        self.device = get_device(device)
        self.precision = precision
        
        # Load model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move to device and set precision
        self.model = self.model.to(self.device)
        if precision == "fp16":
            self.model = self.model.half()
        
        self.model.eval()
        
        # Support set storage
        self.support_embeddings = None
        self.support_labels = None
        self.class_names = None
        
        logger.info(f"Loaded CLIP model: {model_name}")
        logger.info(f"Device: {self.device}, Precision: {precision}")
    
    def fit(self, support_data: List[Dict[str, Any]]) -> None:
        """Fit the model on support set.
        
        Args:
            support_data: List of support examples with 'image', 'text', and 'class_id' keys.
        """
        logger.info(f"Fitting model on {len(support_data)} support examples")
        
        # Extract images and texts
        images = [item["image"] for item in support_data]
        texts = [item["text"] for item in support_data]
        labels = [item["class_id"] for item in support_data]
        
        # Get unique class names
        self.class_names = list(set(texts))
        self.class_names.sort()
        
        # Process images and texts
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = self.model(**inputs)
            image_embeddings = outputs.image_embeds
            text_embeddings = outputs.text_embeds
            
            # Store support embeddings
            self.support_embeddings = {
                "image": image_embeddings,
                "text": text_embeddings,
            }
            self.support_labels = torch.tensor(labels, device=self.device)
        
        logger.info(f"Stored embeddings for {len(self.class_names)} classes")
    
    def predict(self, query_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict on query set.
        
        Args:
            query_data: List of query examples.
            
        Returns:
            Dictionary with predictions and scores.
        """
        if self.support_embeddings is None:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Predicting on {len(query_data)} query examples")
        
        # Extract images and texts
        images = [item["image"] for item in query_data]
        texts = [item["text"] for item in query_data]
        true_labels = [item["class_id"] for item in query_data]
        
        with torch.no_grad():
            # Process query data
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = self.model(**inputs)
            query_image_embeddings = outputs.image_embeds
            query_text_embeddings = outputs.text_embeds
            
            # Calculate similarities
            image_similarities = self._calculate_similarities(
                query_image_embeddings, self.support_embeddings["image"]
            )
            text_similarities = self._calculate_similarities(
                query_text_embeddings, self.support_embeddings["text"]
            )
            
            # Combine similarities (simple average)
            combined_similarities = (image_similarities + text_similarities) / 2
            
            # Get predictions
            predicted_indices = torch.argmax(combined_similarities, dim=1)
            predicted_labels = self.support_labels[predicted_indices]
            
            # Calculate confidence scores
            confidence_scores = torch.max(combined_similarities, dim=1)[0]
            
            # Convert to class names
            predicted_class_names = [self.class_names[i] for i in predicted_indices.cpu().numpy()]
        
        return {
            "predictions": predicted_labels.cpu().numpy(),
            "predicted_class_names": predicted_class_names,
            "confidence_scores": confidence_scores.cpu().numpy(),
            "true_labels": true_labels,
            "similarities": {
                "image": image_similarities.cpu().numpy(),
                "text": text_similarities.cpu().numpy(),
                "combined": combined_similarities.cpu().numpy(),
            },
        }
    
    def _calculate_similarities(
        self, query_embeddings: torch.Tensor, support_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Calculate cosine similarities between query and support embeddings.
        
        Args:
            query_embeddings: Query embeddings.
            support_embeddings: Support embeddings.
            
        Returns:
            Similarity matrix.
        """
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        support_embeddings = F.normalize(support_embeddings, p=2, dim=1)
        
        # Calculate cosine similarities
        similarities = torch.mm(query_embeddings, support_embeddings.t())
        
        return similarities
    
    def evaluate(self, query_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate the model on query set.
        
        Args:
            query_data: List of query examples.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        predictions = self.predict(query_data)
        
        true_labels = torch.tensor(predictions["true_labels"])
        predicted_labels = torch.tensor(predictions["predictions"])
        
        # Calculate accuracy
        accuracy = (predicted_labels == true_labels).float().mean().item()
        
        # Calculate per-class accuracy
        unique_labels = torch.unique(true_labels)
        per_class_accuracy = {}
        
        for label in unique_labels:
            mask = true_labels == label
            if mask.sum() > 0:
                class_acc = (predicted_labels[mask] == true_labels[mask]).float().mean().item()
                per_class_accuracy[f"class_{label.item()}"] = class_acc
        
        # Calculate confidence statistics
        confidence_scores = predictions["confidence_scores"]
        avg_confidence = confidence_scores.mean()
        std_confidence = confidence_scores.std()
        
        return {
            "accuracy": accuracy,
            "per_class_accuracy": per_class_accuracy,
            "avg_confidence": avg_confidence,
            "std_confidence": std_confidence,
            "n_query": len(query_data),
            "n_classes": len(self.class_names),
        }
    
    def zero_shot_predict(
        self, images: List[Any], class_names: List[str]
    ) -> Dict[str, Any]:
        """Perform zero-shot prediction.
        
        Args:
            images: List of images to classify.
            class_names: List of class names.
            
        Returns:
            Dictionary with predictions and scores.
        """
        logger.info(f"Zero-shot prediction on {len(images)} images")
        
        with torch.no_grad():
            # Process images and class names
            inputs = self.processor(
                text=class_names,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = self.model(**inputs)
            image_embeddings = outputs.image_embeds
            text_embeddings = outputs.text_embeds
            
            # Calculate similarities
            similarities = self._calculate_similarities(image_embeddings, text_embeddings)
            
            # Get predictions
            predicted_indices = torch.argmax(similarities, dim=1)
            predicted_class_names = [class_names[i] for i in predicted_indices.cpu().numpy()]
            
            # Calculate confidence scores
            confidence_scores = torch.max(similarities, dim=1)[0]
        
        return {
            "predictions": predicted_indices.cpu().numpy(),
            "predicted_class_names": predicted_class_names,
            "confidence_scores": confidence_scores.cpu().numpy(),
            "similarities": similarities.cpu().numpy(),
        }
    
    def get_embeddings(self, images: List[Any], texts: List[str]) -> Dict[str, torch.Tensor]:
        """Get embeddings for images and texts.
        
        Args:
            images: List of images.
            texts: List of texts.
            
        Returns:
            Dictionary with image and text embeddings.
        """
        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            return {
                "image_embeddings": outputs.image_embeds,
                "text_embeddings": outputs.text_embeds,
            }
