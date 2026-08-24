import torch
import torch.nn.functional as F

class KnowledgeDistillationTrainer:
    """
    A robust trainer class for performing Knowledge Distillation with built-in validation.

    Knowledge Distillation transfers the learned behavior of a large, complex model
    (the Teacher) to a smaller, more efficient model (the Student). This process not
    only improves the Student's accuracy compared to training from scratch, but also
    enhances its interpretability by transferring class-similarity information[cite: 1, 2].
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        optimizer,
        device,
        val_loader=None,
        temperature=4.0,
        alpha=0.5
    ):
        """
        Initializes the KnowledgeDistillationTrainer with all required components.

        Arguments:
        ----------
        teacher_model : transformers.PreTrainedModel
            The pre-trained, fine-tuned expert model. Its weights are FROZEN.

        student_model : transformers.PreTrainedModel
            The lightweight model that will be trained to mimic the teacher.

        train_loader : torch.utils.data.DataLoader
            DataLoader providing batches of training data.

        optimizer : torch.optim.Optimizer
            The optimization algorithm (e.g., AdamW) attached ONLY to the student's parameters.

        device : torch.device
            The hardware accelerator to use ('cuda' or 'cpu').

        val_loader : torch.utils.data.DataLoader, optional
            DataLoader providing batches of validation data to evaluate the model
            at the end of each epoch.

        temperature : float, optional (default=4.0)
            Controls the softness of the probability distribution. A higher temperature
            reveals "dark knowledge" (class-similarity) from the teacher[cite: 1, 2].

        alpha : float, optional (default=0.5)
            The balancing weight between the true labels (0.0) and Teacher's advice (1.0).
        """
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        self.teacher_model = teacher_model.to(self.device)
        self.student_model = student_model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer

        # Ensure the teacher is strictly in inference mode
        self.teacher_model.eval()

    def _compute_loss(self, student_logits, teacher_logits, labels):
        """
        Calculates the combined Cross-Entropy and Knowledge Distillation loss.
        """
        # Standard Cross-Entropy Loss
        loss_ce = F.cross_entropy(student_logits, labels)

        # Distillation Loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        loss_distillation = F.kl_div(student_log_probs, soft_targets, reduction='batchmean')

        # Scale by T^2 to balance the gradient magnitudes
        loss_distillation = loss_distillation * (self.temperature ** 2)

        # Combined Total Loss
        total_loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_distillation

        return total_loss

    def evaluate(self):
        """
        Evaluates the student model's performance on the validation dataset.

        Returns:
        --------
        val_loss : float
            The average Cross-Entropy loss on the validation set.
        accuracy : float
            The percentage of correctly classified images (from 0.0 to 1.0).
        """
        # Set the student to evaluation mode (disables Dropout/BatchNorm updates)
        self.student_model.eval()

        running_val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # We don't need gradients for validation, saving memory and compute
        with torch.no_grad():
            for batch in self.val_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get student predictions
                student_outputs = self.student_model(pixel_values=pixel_values)
                student_logits = student_outputs.logits

                # Calculate standard Cross-Entropy loss for validation
                # (We don't use distillation loss here, just real-world performance)
                loss = F.cross_entropy(student_logits, labels)
                running_val_loss += loss.item()

                # Calculate accuracy: find the index with the highest probability
                predictions = torch.argmax(student_logits, dim=-1)

                # Count how many predictions match the true labels
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        # Calculate final metrics
        avg_val_loss = running_val_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples

        return avg_val_loss, accuracy

    def train(self, epochs):
        """
        Executes the training and validation loop for the given number of epochs.
        """
        print(f"Starting Knowledge Distillation on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self.student_model.train() # Set back to train mode!
            running_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.optimizer.zero_grad()

                # Teacher makes soft predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(pixel_values=pixel_values)
                    teacher_logits = teacher_outputs.logits

                # Student makes predictions
                student_outputs = self.student_model(pixel_values=pixel_values)
                student_logits = student_outputs.logits

                # Calculate loss and update weights
                loss = self._compute_loss(student_logits, teacher_logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(self.train_loader)

            # --- VALIDATION PHASE ---
            if self.val_loader is not None:
                # Call our new evaluation method
                val_loss, val_acc = self.evaluate()

                # Print comprehensive epoch summary
                print(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Accuracy: {val_acc*100:.2f}%")
            else:
                # Print basic summary if no validation data is provided
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}")

        print("Training complete! The student has been successfully distilled.")

    def save_student(self, save_path):
        """
        Saves the trained student model's weights and configuration to disk.
        """
        self.student_model.save_pretrained(save_path)
        print(f"Student model successfully saved to '{save_path}'")