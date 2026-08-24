import os
import glob
import math
import torch
from transformers import AutoModelForImageClassification
import json

class TracInExplainer:
    """
    A class to compute training data influence for Vision Transformers (ViT) 
    using the TracInCP method with Random Projections.
    """
    
    def __init__(self, checkpoints_dir, learning_rate=5e-5, projection_dim=1000, device="cuda"):
        """
        Initializes the TracInExplainer.
        
        Args:
            checkpoints_dir (str): Path to the directory containing model checkpoints.
            learning_rate (float): The learning rate used during training. Default is 5e-5.
            projection_dim (int): The reduced dimension for random projections to save memory. Default is 1000.
            device (str): The device to run computations on ('cuda' or 'cpu'). Default is 'cuda'.
        """
        self.checkpoints_dir = checkpoints_dir
        self.learning_rate = learning_rate
        self.projection_dim = projection_dim
        self.device = device
        
        self.checkpoints = self._get_sorted_checkpoints()
        self.projection_matrix = None
        self.train_database = None
        
        if not self.checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoints_dir}")

    def _get_sorted_checkpoints(self):
        """
        Finds and sorts checkpoint folders chronologically based on their step number.
        
        Returns:
            list: A list of sorted checkpoint directory paths.
        """
        pattern = os.path.join(self.checkpoints_dir, "checkpoint-*")
        checkpoints = glob.glob(pattern)
        # Sort mathematically by the step number (e.g., checkpoint-500 < checkpoint-1000)
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        return checkpoints

    def _initialize_projection_matrix(self):
        """
        Creates the random projection matrix (G) to flatten and compress the gradients.
        This matrix must remain constant for both training and testing phases.
        """
        # Load the first checkpoint temporarily just to get the classifier's layer dimensions
        temp_model = AutoModelForImageClassification.from_pretrained(self.checkpoints[0])
        num_parameters = temp_model.classifier.weight.numel()
        del temp_model
        
        # Standard deviation for the normal distribution: sqrt(1/d)
        std_dev = math.sqrt(1.0 / self.projection_dim)
        
        self.projection_matrix = torch.randn(self.projection_dim, num_parameters) * std_dev
        self.projection_matrix = self.projection_matrix.to(self.device)

    def prepare_database(self, train_dataloader, image_names, save_path="tracin_db.pt"):
        """
        Iterates through all checkpoints and training data to pre-compute 
        and save the flattened, projected gradients.
        
        Args:
            train_dataloader (DataLoader): PyTorch DataLoader with the training data. 
                                           CRITICAL: Must be instantiated with shuffle=False.
            image_names (list): A list of strings containing the file names/paths of the 
                                training images in the exact same order as the dataloader.
            save_path (str): File path to save the resulting database dictionary. Default is 'tracin_db.pt'.
        """
        print(f"Preparing database using {len(self.checkpoints)} checkpoints...")
        
        # Security check to ensure we have exactly one name per image
        total_images = sum(batch['pixel_values'].size(0) for batch in train_dataloader)
        if len(image_names) != total_images:
            raise ValueError(f"Mismatch: DataLoader has {total_images} images, but {len(image_names)} names were provided.")
        
        if self.projection_matrix is None:
            self._initialize_projection_matrix()
            
        self.train_database = {}
        global_image_id = 0
        
        for ckpt_idx, ckpt_path in enumerate(self.checkpoints):
            print(f"Processing {os.path.basename(ckpt_path)}...")
            
            model = AutoModelForImageClassification.from_pretrained(ckpt_path)
            model.to(self.device)
            model.eval()
            
            # Reset global ID for each checkpoint to align the names correctly
            global_image_id = 0 
            
            for batch in train_dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Process sample by sample within the batch
                for i in range(pixel_values.size(0)):
                    single_input = pixel_values[i].unsqueeze(0)
                    single_label = labels[i].unsqueeze(0)
                    
                    current_image_name = image_names[global_image_id]
                    
                    model.zero_grad()
                    outputs = model(pixel_values=single_input, labels=single_label)
                    loss = outputs.loss
                    loss.backward()
                    
                    # Extract, flatten, and project
                    giant_grad = model.classifier.weight.grad.view(-1)
                    projected_grad = torch.matmul(self.projection_matrix, giant_grad)
                    
                    # Store using the image name as the key
                    if current_image_name not in self.train_database:
                        self.train_database[current_image_name] = []
                        
                    self.train_database[current_image_name].append(projected_grad.cpu())
                    global_image_id += 1
                    
            del model
            torch.cuda.empty_cache()
            
        # Save the database and the projection matrix
        save_data = {
            "matrix_G": self.projection_matrix.cpu(),
            "train_db": self.train_database
        }
        torch.save(save_data, save_path)
        print(f"Database successfully saved to {save_path}.")

    def load_database(self, load_path="tracin_db.pt"):
        """
        Loads a previously computed training database from disk to skip the preparation step.
        
        Args:
            load_path (str): The file path to the saved database. Default is 'tracin_db.pt'.
        """
        print(f"Loading database from {load_path}...")
        data = torch.load(load_path)
        self.projection_matrix = data["matrix_G"].to(self.device)
        self.train_database = data["train_db"]
        print("Database loaded successfully.")

    def generate(self, test_pixel_values, test_label=None, top_k=5):
        """
        Analyzes a single test image and computes the influence scores of training samples.
        
        Args:
            test_pixel_values (torch.Tensor): The processed image tensor. Shape: (1, C, H, W).
            test_label (torch.Tensor, optional): The ground truth label. If None, the model's 
                                                 own predicted class will be used. Default is None.
            top_k (int): Number of top proponents and opponents to return. Default is 5.
            
        Returns:
            dict: A dictionary containing the explained label, the top proponents, and top opponents.
        """
        if self.train_database is None or self.projection_matrix is None:
            raise RuntimeError("Database not loaded. Call load_database() or prepare_database() first.")
            
        test_pixel_values = test_pixel_values.to(self.device)
        
        # Determine the label to explain
        if test_label is None:
            # Load the LAST checkpoint to get the final model's prediction
            print("No label provided. Getting the model's prediction...")
            final_model = AutoModelForImageClassification.from_pretrained(self.checkpoints[-1]).to(self.device)
            final_model.eval()
            
            with torch.no_grad():
                logits = final_model(pixel_values=test_pixel_values).logits
                predicted_class = torch.argmax(logits, dim=1)
                
            test_label = predicted_class
            print(f"Explaining the predicted class ID: {test_label.item()}")
            
            del final_model
            torch.cuda.empty_cache()
        else:
            test_label = test_label.to(self.device)
            print(f"Explaining the provided Ground Truth class ID: {test_label.item()}")
            
        test_vectors = []
        
        # 1. Compute projected gradients for the test image across all checkpoints
        for ckpt_path in self.checkpoints:
            model = AutoModelForImageClassification.from_pretrained(ckpt_path)
            model.to(self.device)
            model.eval()
            
            model.zero_grad()
            outputs = model(pixel_values=test_pixel_values, labels=test_label)
            loss = outputs.loss
            loss.backward()
            
            giant_grad = model.classifier.weight.grad.view(-1)
            projected_grad = torch.matmul(self.projection_matrix, giant_grad)
            test_vectors.append(projected_grad.cpu())
            
            del model
            torch.cuda.empty_cache()
            
        # Compute influence scores (TracInCP formula)
        influence_scores = {}
        
        for train_id, train_vectors in self.train_database.items():
            total_score = 0.0
            for ckpt_idx in range(len(self.checkpoints)):
                vec_train = train_vectors[ckpt_idx]
                vec_test = test_vectors[ckpt_idx]
                
                # Dot product multiplied by learning rate
                dot_product = torch.dot(vec_train, vec_test).item()
                total_score += dot_product * self.learning_rate
                
            influence_scores[train_id] = total_score
            
        # Sort and extract top proponents and opponents
        sorted_scores = sorted(influence_scores.items(), key=lambda item: item[1], reverse=True)
        
        proponents = sorted_scores[:top_k]
        opponents = sorted_scores[-top_k:]
        opponents.reverse() 
        
        return {
            "explained_label": test_label.item(),
            "proponents": proponents,
            "opponents": opponents
        }

    def evaluate_test_dataset(self, test_dataloader, test_image_names, top_k=5, save_path="test_results.json", explain_prediction=True):
        """
        Efficiently computes TracInCP influence scores for an entire test dataset.
        Loads each checkpoint only once to save time and avoids redundant I/O operations.
        
        Args:
            test_dataloader (DataLoader): PyTorch DataLoader with the test data (shuffle=False).
            test_image_names (list): A list of test image names matching the dataloader order.
            top_k (int): Number of top proponents and opponents to save per test image. Default is 5.
            save_path (str): Path to save the resulting JSON file. Default is 'test_results.json'.
            explain_prediction (bool): If True, explains the model's predictions. 
                                       If False, explains the ground truth labels. Default is False.
                                       
        Returns:
            dict: A dictionary with test image names as keys and their proponents/opponents as values.
        """
        if self.train_database is None or self.projection_matrix is None:
            raise RuntimeError("Database not loaded. Call load_database() or prepare_database() first.")
            
        print(f"Step 1: Extracting test gradients across {len(self.checkpoints)} checkpoints...")
        
        # Initialize dictionary to store vectors for each test image
        test_vectors = {name: [] for name in test_image_names}
        
        for ckpt_path in self.checkpoints:
            print(f"Loading {os.path.basename(ckpt_path)}...")
            model = AutoModelForImageClassification.from_pretrained(ckpt_path).to(self.device)
            model.eval()
            
            global_test_id = 0
            for batch in test_dataloader:
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Decide which label to explain
                if explain_prediction:
                    # Calculate the model's prediction on the fly
                    with torch.no_grad():
                        logits = model(pixel_values=pixel_values).logits
                        target_labels = torch.argmax(logits, dim=1)
                else:
                    # Use the ground truth from the dataloader
                    target_labels = batch['labels'].to(self.device)
                
                # Process sample by sample
                for i in range(pixel_values.size(0)):
                    current_name = test_image_names[global_test_id]
                    single_input = pixel_values[i].unsqueeze(0)
                    single_label = target_labels[i].unsqueeze(0)
                    
                    model.zero_grad()
                    outputs = model(pixel_values=single_input, labels=single_label)
                    loss = outputs.loss
                    loss.backward()
                    
                    giant_grad = model.classifier.weight.grad.view(-1)
                    projected_grad = torch.matmul(self.projection_matrix, giant_grad)
                    
                    test_vectors[current_name].append(projected_grad.cpu())
                    global_test_id += 1
            
            del model
            torch.cuda.empty_cache()
            
        print("\nStep 2: Computing influence scores against training database (This is fast!)...")
        all_results = {}
        
        for test_name, test_vecs in test_vectors.items():
            influence_scores = {}
            
            # Compare current test image against ALL training images
            for train_name, train_vecs in self.train_database.items():
                total_score = 0.0
                for ckpt_idx in range(len(self.checkpoints)):
                    dot_product = torch.dot(train_vecs[ckpt_idx], test_vecs[ckpt_idx]).item()
                    total_score += dot_product * self.learning_rate
                influence_scores[train_name] = total_score
                
            # Sort and extract top K
            sorted_scores = sorted(influence_scores.items(), key=lambda item: item[1], reverse=True)
            proponents = sorted_scores[:top_k]
            opponents = sorted_scores[-top_k:]
            opponents.reverse()
            
            all_results[test_name] = {
                "proponents": proponents,
                "opponents": opponents
            }
            
        # Save to JSON for easy analysis later
        with open(save_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)
            
        print(f"\nDone! Results for {len(test_image_names)} test images saved to {save_path}")
        return all_results