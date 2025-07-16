// Application state
let currentStep = 1;
let totalSteps = 6;
let uploadedPhotos = [];
let petData = {};
let savedPets = JSON.parse(localStorage.getItem('savedPets')) || [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeApplication();
});

/**
 * Initializes the application by setting up UI components and event handlers
 */
function initializeApplication() {
    // Load saved pets and update dashboard
    updateDashboard();

    // Populate dropdowns
    populateBreedDropdown();
    populateColorDropdown();
    populateStateDropdown();

    // Set up event listeners
    setupEventListeners();

    // Ensure dashboard is shown by default (with small delay to ensure DOM is ready)
    setTimeout(() => {
        showDashboard();
    }, 50);
}

/**
 * Sets up all event listeners for form interactions and navigation
 */
function setupEventListeners() {
    // Photo upload handling
    const photoInput = document.getElementById('photo-input');
    const photoUploadArea = document.getElementById('photo-upload-area');

    if (photoInput) {
        photoInput.addEventListener('change', handlePhotoSelection);
    }

    if (photoUploadArea) {
        photoUploadArea.addEventListener('click', () => photoInput.click());
        photoUploadArea.addEventListener('dragover', handleDragOver);
        photoUploadArea.addEventListener('drop', handleDrop);
        photoUploadArea.addEventListener('dragenter', handleDragEnter);
        photoUploadArea.addEventListener('dragleave', handleDragLeave);
    }

    // Form validation
    const form = document.getElementById('pet-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    // Pet type change handler
    const petTypeSelect = document.getElementById('pet-type');
    if (petTypeSelect) {
        petTypeSelect.addEventListener('change', handlePetTypeChange);
    }

    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const target = this.getAttribute('href').substring(1);
            if (target === 'dashboard') {
                showDashboard();
            } else if (target === 'add-pet') {
                showAddPetForm();
            }
        });
    });

    // View All link
    const viewAllLink = document.querySelector('.view-all-link');
    if (viewAllLink) {
        viewAllLink.addEventListener('click', function (e) {
            e.preventDefault();
            showAllPets();
        });
    }
}

/**
 * Shows the dashboard view and updates navigation state
 */
function showDashboard() {
    document.getElementById('dashboard').style.display = 'block';
    document.getElementById('add-pet').style.display = 'none';

    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    document.querySelector('.nav-link[href="#dashboard"]').classList.add('active');

    updateDashboard();
}

/**
 * Shows the add pet form and updates navigation state
 */
function showAddPetForm() {
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('add-pet').style.display = 'block';

    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    document.querySelector('.nav-link[href="#add-pet"]').classList.add('active');

    // Reset form if not editing
    if (!petData.editingId) {
        resetForm();
    }
}

/**
 * Resets the form to its initial state
 */
function resetForm() {
    // Reset form data
    currentStep = 1;
    uploadedPhotos = [];
    petData = {};

    // Reset form fields
    const form = document.getElementById('pet-form');
    if (form) {
        form.reset();
    }

    // Reset form title
    const formTitle = document.querySelector('.section-title');
    if (formTitle) {
        formTitle.textContent = 'Add New Pet Profile';
    }

    // Update step display
    updateStepDisplay();

    // Clear photo preview
    updatePhotoPreview();

    // Reset breed dropdown
    populateBreedDropdown();
}

/**
 * Updates the dashboard with current statistics and pet data
 */
function updateDashboard() {
    const totalPets = savedPets.length;
    const avgPrediction = calculateAveragePrediction();
    const fastAdopters = countFastAdopters();
    const avgConfidence = calculateAverageConfidence();

    document.getElementById('total-pets').textContent = totalPets;
    document.getElementById('pets-in-system').textContent = `${totalPets} pets in system`;
    document.getElementById('avg-prediction').textContent = avgPrediction;
    document.getElementById('fast-adopters').textContent = fastAdopters;
    document.getElementById('fast-prediction').textContent = '≤ 14 days predicted';
    document.getElementById('avg-confidence').textContent = avgConfidence;

    // Update recent pets
    updateRecentPets();
}

/**
 * Calculates the average prediction time for all pets
 * @returns {string} Average prediction in days
 */
function calculateAveragePrediction() {
    if (savedPets.length === 0) return '-- days';

    const predictions = savedPets.map(pet => {
        switch (pet.prediction) {
            case 0: return 3.5;     // 0-7 days average
            case 1: return 19;      // 8-30 days average
            case 2: return 60;      // 31-90 days average
            case 3: return 120;     // 100+ days average
            default: return 30;
        }
    });

    const avg = predictions.reduce((sum, days) => sum + days, 0) / predictions.length;
    return Math.round(avg) + ' days';
}

/**
 * Counts pets with fast adoption predictions (≤30 days)
 * @returns {number} Number of fast adopters
 */
function countFastAdopters() {
    return savedPets.filter(pet => pet.prediction <= 1).length;
}

/**
 * Calculates the average confidence level for all pets
 * @returns {string} Average confidence percentage
 */
function calculateAverageConfidence() {
    if (savedPets.length === 0) return '--%';

    const confidences = savedPets.map(pet => pet.confidence || 75);
    const avg = confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length;
    return Math.round(avg) + '%';
}

/**
 * Updates the recent pets grid with the latest 6 pets
 */
function updateRecentPets() {
    const recentPetsGrid = document.getElementById('recent-pets-grid');
    const recentPets = savedPets.slice(-6).reverse();

    recentPetsGrid.innerHTML = '';

    recentPets.forEach(pet => {
        const petCard = createPetCard(pet);
        recentPetsGrid.appendChild(petCard);
    });
}

/**
 * Creates a pet card element for display in the dashboard
 * @param {Object} pet - Pet object with all pet data
 * @returns {HTMLElement} Pet card DOM element
 */
function createPetCard(pet) {
    const card = document.createElement('div');
    card.className = 'pet-card';

    const period = ADOPTION_PERIODS[pet.prediction] || ADOPTION_PERIODS[2];
    const breedName = getBreedName(pet.type, pet.breed);

    card.innerHTML = `
        <div class="pet-card-header">
            <button class="delete-pet-btn" onclick="deletePet('${pet.id}')" title="Delete Pet">
                <i class="fas fa-trash"></i>
            </button>
        </div>
        <img src="${pet.photos && pet.photos.length > 0 ? pet.photos[0] : 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200"><rect width="300" height="200" fill="%23f1f5f9"/><text x="150" y="100" text-anchor="middle" fill="%2364748b" font-family="Arial" font-size="14">No Image</text></svg>'}" 
             alt="${pet.name || 'Pet'}" class="pet-image">
        <div class="pet-info">
            <div class="pet-name">${pet.name || 'Unnamed'}</div>
            <div class="pet-type-badge">${getTypeName(pet.type)}</div>
            <div class="pet-details">
                <div class="pet-detail">
                    <i class="fas fa-calendar-plus"></i>
                    <span>Arrived: ${formatDate(pet.dateAdded)}</span>
                </div>
                <div class="pet-detail">
                    <i class="fas fa-clock"></i>
                    <span>Predicted: ${period.days}</span>
                </div>
                <div class="pet-detail">
                    <i class="fas fa-heart"></i>
                    <span>${breedName}</span>
                </div>
            </div>
            <div class="pet-footer">
                <div class="pet-prediction">
                    <span class="prediction-badge ${period.badge}">${period.badge}</span>
                </div>
                <div class="pet-fee">$${parseFloat(pet.fee || 0).toFixed(0)}</div>
            </div>
        </div>
    `;

    // Make the entire card clickable, but prevent clicks on delete button
    card.style.cursor = 'pointer';
    card.addEventListener('click', (e) => {
        // Don't open modal if delete button or its icon was clicked
        if (e.target.closest('.delete-pet-btn')) {
            return;
        }
        showPetDetails(pet.id);
    });

    return card;
}

/**
 * Gets the breed name from type and breed ID
 * @param {number} type - Pet type ID
 * @param {number} breedId - Breed ID
 * @returns {string} Breed name
 */
function getBreedName(type, breedId) {
    if (!type || !breedId) return 'Unknown Breed';
    return BREED_DATA[type] && BREED_DATA[type][breedId] ? BREED_DATA[type][breedId] : 'Unknown Breed';
}

/**
 * Converts pet type ID to readable name
 * @param {number} typeId - Pet type ID
 * @returns {string} Pet type name
 */
function getTypeName(typeId) {
    const typeMap = {
        1: 'Dog',
        2: 'Cat',
    };
    return typeMap[typeId] || 'Unknown';
}

/**
 * Converts gender ID to readable name
 * @param {number} genderId - Gender ID
 * @returns {string} Gender name
 */
function getGenderName(genderId) {
    const genderMap = {
        1: 'Male',
        2: 'Female',
        3: 'Mixed'
    };
    return genderMap[genderId] || 'Unknown';
}

/**
 * Gets color name from color ID
 * @param {number} colorId - Color ID
 * @returns {string} Color name
 */
function getColorName(colorId) {
    return COLOR_DATA[colorId] || 'Unknown Color';
}

/**
 * Converts maturity size ID to readable name
 * @param {number} sizeId - Size ID
 * @returns {string} Size name
 */
function getMaturitySizeName(sizeId) {
    const sizeMap = {
        1: 'Small',
        2: 'Medium',
        3: 'Large',
        4: 'Extra Large',
        0: 'Not Specified'
    };
    return sizeMap[sizeId] || 'Unknown';
}

/**
 * Converts fur length ID to readable name
 * @param {number} furId - Fur length ID
 * @returns {string} Fur length name
 */
function getFurLengthName(furId) {
    const furMap = {
        1: 'Short',
        2: 'Medium',
        3: 'Long',
        0: 'Not Specified'
    };
    return furMap[furId] || 'Unknown';
}

/**
 * Converts vaccination status ID to readable text
 * @param {number} status - Vaccination status ID
 * @returns {string} Vaccination status
 */
function getVaccinationStatus(status) {
    const statusMap = {
        1: 'Yes',
        2: 'No',
        3: 'Not Sure'
    };
    return statusMap[status] || 'Unknown';
}

/**
 * Converts health status ID to readable text
 * @param {number} healthId - Health status ID
 * @returns {string} Health status
 */
function getHealthStatus(healthId) {
    const healthMap = {
        1: 'Healthy',
        2: 'Minor Injury',
        3: 'Serious Injury',
        4: 'Not Specified'
    };
    return healthMap[healthId] || 'Unknown';
}

/**
 * Populates the breed dropdown (initially empty until pet type is selected)
 */
function populateBreedDropdown() {
    const breedSelect = document.getElementById('pet-breed');
    if (!breedSelect) return;

    // Will be populated when pet type is selected
    breedSelect.innerHTML = '<option value="">First select pet type</option>';
}

/**
 * Handles pet type selection and updates breed dropdown accordingly
 */
function handlePetTypeChange() {
    const petType = document.getElementById('pet-type').value;
    const breedSelect = document.getElementById('pet-breed');

    if (!petType || !breedSelect) return;

    // Clear existing options
    breedSelect.innerHTML = '<option value="">Select breed</option>';

    // Populate with breeds for selected type
    const breeds = BREED_DATA[petType];
    if (breeds) {
        Object.entries(breeds).forEach(([id, name]) => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = name;
            breedSelect.appendChild(option);
        });
    }
}

/**
 * Populates the color dropdown with available colors
 */
function populateColorDropdown() {
    const colorSelect = document.getElementById('pet-color');
    if (!colorSelect) return;

    colorSelect.innerHTML = '<option value="">Select color</option>';

    Object.entries(COLOR_DATA).forEach(([id, name]) => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = name;
        colorSelect.appendChild(option);
    });
}

/**
 * Populates the state dropdown with available states
 */
function populateStateDropdown() {
    const stateSelect = document.getElementById('pet-state');
    if (!stateSelect) return;

    stateSelect.innerHTML = '<option value="">Select state</option>';

    Object.entries(STATE_DATA).forEach(([id, name]) => {
        const option = document.createElement('option');
        option.value = id;
        option.textContent = name;
        stateSelect.appendChild(option);
    });
}

/**
 * Handles form submission, validates current step, and saves pet data
 */
function nextStep() {
    if (currentStep < totalSteps) {
        currentStep++;
        updateStepDisplay();
    }
}

/**
 * Handles form submission, validates current step, and saves pet data
 */
function previousStep() {
    if (currentStep > 1) {
        currentStep--;
        updateStepDisplay();
    }
}

/**
 * Handles form submission, validates current step, and saves pet data
 */
function showStep(step) {
    // Hide all steps
    document.querySelectorAll('.form-step').forEach(stepEl => {
        stepEl.classList.remove('active');
    });

    // Show current step
    const currentStepEl = document.getElementById(`step-${step}`);
    if (currentStepEl) {
        currentStepEl.classList.add('active');
    }

    // Update step indicators
    document.querySelectorAll('.step').forEach((stepEl, index) => {
        stepEl.classList.remove('active', 'completed');
        if (index + 1 === step) {
            stepEl.classList.add('active');
        } else if (index + 1 < step) {
            stepEl.classList.add('completed');
        }
    });

    // Update navigation buttons
    updateNavigationButtons();
}

/**
 * Handles form submission, validates current step, and saves pet data
 */
function updateProgress() {
    const progressPercentage = (currentStep / totalSteps) * 100;
    const progressFill = document.getElementById('progress-fill');
    const currentStepSpan = document.getElementById('current-step');
    const completionPercentage = document.getElementById('completion-percentage');

    if (progressFill) {
        progressFill.style.width = `${progressPercentage}%`;
    }

    if (currentStepSpan) {
        currentStepSpan.textContent = currentStep;
    }

    if (completionPercentage) {
        completionPercentage.textContent = `${Math.round(progressPercentage)}%`;
    }
}

/**
 * Handles form submission, validates current step, and saves pet data
 */
function updateStepDisplay() {
    // Update step indicator
    document.getElementById('current-step').textContent = currentStep;

    // Update progress percentage
    const percentage = Math.round((currentStep / totalSteps) * 100);
    document.getElementById('completion-percentage').textContent = percentage + '%';
    document.getElementById('progress-fill').style.width = percentage + '%';

    // Update step classes
    document.querySelectorAll('.step').forEach((step, index) => {
        const stepNumber = index + 1;
        step.classList.remove('active', 'completed');

        if (stepNumber === currentStep) {
            step.classList.add('active');
        } else if (stepNumber < currentStep) {
            step.classList.add('completed');
        }
    });

    // Show/hide form steps
    document.querySelectorAll('.form-step').forEach((step, index) => {
        step.classList.remove('active');
        if (index + 1 === currentStep) {
            step.classList.add('active');
        }
    });

    // Update navigation buttons
    updateNavigationButtons();
}

/**
 * Handles form submission, validates current step, and saves pet data
 */
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const submitBtn = document.getElementById('submit-btn');

    if (prevBtn) {
        prevBtn.style.display = currentStep > 1 ? 'inline-flex' : 'none';
    }

    if (nextBtn) {
        nextBtn.style.display = currentStep < totalSteps ? 'inline-flex' : 'none';
    }

    if (submitBtn) {
        submitBtn.style.display = currentStep === totalSteps ? 'inline-flex' : 'none';
    }
}

/**
 * Handles form submission, validates current step, and saves pet data
 * @param {Event} event - Form submission event
 * @return {boolean} True if submission is valid, false otherwise
 */
function validateCurrentStep() {
    const currentStepEl = document.getElementById(`step-${currentStep}`);
    if (!currentStepEl) return false;

    const requiredFields = currentStepEl.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            isValid = false;
            field.focus();
            field.style.borderColor = '#ef4444';

            // Reset border color after a delay
            setTimeout(() => {
                field.style.borderColor = '';
            }, 3000);
        }
    });

    return isValid;
}

/**
 * Handles form submission, validates current step, and saves pet data
 * @param {Event} event - Form submission event
 * @return {boolean} True if submission is valid, false otherwise
 */
function handlePhotoSelection(event) {
    const files = Array.from(event.target.files);
    processFiles(files);
}

/**
 * Handles drag and drop events for photo upload area
 * @param {Event} event - Drag and drop event
 * @return {void}
 */
function handleDragOver(event) {
    event.preventDefault();
}

/**
 * Handles drag enter event for photo upload area
 * @param {Event} event - Drag enter event
 * @return {void}
 */
function handleDragEnter(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

/**
 * Handles drag leave event for photo upload area
 * @param {Event} event - Drag leave event
 * @return {void}
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
}

/**
 * Handles drop event for photo upload area
 * @param {Event} event - Drop event
 * @return {void}
 */
function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');

    const files = Array.from(event.dataTransfer.files);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    processFiles(imageFiles);
}

/**
 * Processes selected files and updates the photo preview
 * @param {File[]} files - Array of File objects
 * @return {void}
 */
function processFiles(files) {
    files.forEach(file => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function (e) {
                uploadedPhotos.push({
                    file: file,
                    dataUrl: e.target.result,
                    name: file.name
                });
                updatePhotoPreview();
            };
            reader.readAsDataURL(file);
        }
    });
}

/**
 * Updates the photo preview section with uploaded images
 */
function updatePhotoPreview() {
    const photoPreview = document.getElementById('photo-preview');
    const uploadWarning = document.getElementById('upload-warning');

    photoPreview.innerHTML = '';

    uploadedPhotos.forEach((photo, index) => {
        const previewItem = document.createElement('div');
        previewItem.className = 'photo-preview-item';
        previewItem.innerHTML = `
            <img src="${photo.dataUrl}" alt="${photo.name}">
            <button type="button" class="photo-remove" onclick="removePhoto(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        photoPreview.appendChild(previewItem);
    });

    // Show/hide warning
    if (uploadWarning) {
        uploadWarning.style.display = uploadedPhotos.length === 0 ? 'block' : 'none';
    }
}

/**
 * Removes a photo from the uploaded photos array and updates the preview
 * @param {number} index - Index of the photo to remove
 * @return {void}
 */
function removePhoto(index) {
    uploadedPhotos.splice(index, 1);
    updatePhotoPreview();
}

/**
 * Handles form submission, validates current step, and saves pet data
 * @param {Event} event - Form submission event
 * @return {boolean} True if submission is valid, false otherwise
 */
function generatePrediction() {
    // Collect form data
    collectFormData();

    // Simulate ML prediction
    const prediction = simulatePrediction(petData);

    // Show prediction modal
    showPredictionModal(prediction);
}

/**
 * Collects form data and prepares the petData object
 */
function collectFormData() {
    const formData = new FormData(document.getElementById('pet-form'));

    petData = {
        id: Date.now().toString(),
        type: parseInt(formData.get('type')),
        name: formData.get('name') || '',
        age: parseInt(formData.get('age')),
        breed: parseInt(formData.get('breed')),
        gender: parseInt(formData.get('gender')),
        color: parseInt(formData.get('color')),
        maturitySize: parseInt(formData.get('maturitySize')),
        furLength: parseInt(formData.get('furLength')),
        vaccinated: parseInt(formData.get('vaccinated')),
        dewormed: parseInt(formData.get('dewormed')),
        sterilized: parseInt(formData.get('sterilized')),
        health: parseInt(formData.get('health')),
        quantity: parseInt(formData.get('quantity')),
        fee: parseFloat(formData.get('fee')),
        state: parseInt(formData.get('state')),
        description: formData.get('description') || '',
        photos: uploadedPhotos.map(photo => photo.dataUrl),
        photoAmt: uploadedPhotos.length,
        rescuerID: 1, // Default rescuer ID
        dateAdded: new Date().toISOString()
    };
}

/**
 * Simulates ML prediction using rule-based scoring
 * @param {Object} data - Pet data object
 * @returns {Object} Prediction result with class, confidence, and score
 */
function simulatePrediction(data) {
    // Simple rule-based prediction simulation
    // In a real application, this would call your ML model

    let score = 0;

    // Age factor (younger pets adopted faster)
    if (data.age <= 6) score += 30;
    else if (data.age <= 24) score += 20;
    else if (data.age <= 60) score += 10;
    else score += 0;

    // Breed factor (popular breeds adopted faster)
    const popularDogBreeds = [20, 109, 141, 103, 39]; // Beagle, Golden Retriever, Labrador, German Shepherd, Border Collie
    const popularCatBreeds = [244, 264, 266, 268]; // Domestic Shorthair, Ragdoll, Scottish Fold, Siamese

    if (data.type === 1 && popularDogBreeds.includes(data.breed)) score += 20;
    if (data.type === 2 && popularCatBreeds.includes(data.breed)) score += 20;
    if (data.breed === 307) score += 10; // Mixed breed bonus

    // Health factors
    if (data.vaccinated === 1) score += 15;
    if (data.dewormed === 1) score += 10;
    if (data.sterilized === 1) score += 10;
    if (data.health === 1) score += 15;

    // Size factor (smaller pets often adopted faster)
    if (data.maturitySize === 1) score += 15;
    else if (data.maturitySize === 2) score += 10;
    else if (data.maturitySize === 3) score += 5;

    // Fee factor (lower fees = faster adoption)
    if (data.fee === 0) score += 20;
    else if (data.fee <= 50) score += 15;
    else if (data.fee <= 100) score += 10;
    else if (data.fee <= 200) score += 5;

    // Photo factor
    if (data.photoAmt >= 3) score += 15;
    else if (data.photoAmt >= 1) score += 10;

    // Description factor
    if (data.description && data.description.length > 50) score += 10;

    // Add random variation to make predictions more realistic
    const randomVariation = Math.random() * 10 - 5; // -5 to +5
    score += randomVariation;

    // Determine prediction based on score
    let predictionClass;
    let confidence;

    if (score >= 80) {
        predictionClass = 0; // Same day - 1 week
        confidence = Math.min(95, 75 + Math.random() * 20);
    } else if (score >= 60) {
        predictionClass = 1; // 1 week - 1 month  
        confidence = Math.min(90, 70 + Math.random() * 20);
    } else if (score >= 40) {
        predictionClass = 2; // 1-3 months
        confidence = Math.min(85, 65 + Math.random() * 20);
    } else {
        predictionClass = 3; // 3+ months
        confidence = Math.min(80, 60 + Math.random() * 20);
    }

    return {
        class: predictionClass,
        confidence: Math.round(confidence),
        score: score
    };
}

/**
 * Shows the prediction modal with the result from the ML model
 * @param {Object} prediction - Prediction result object
 * @return {void}
 */
function showPredictionModal(prediction) {
    const modal = document.getElementById('prediction-modal');
    const predictionTitle = document.getElementById('prediction-title');
    const predictionDescription = document.getElementById('prediction-description');
    const confidenceLevel = document.getElementById('confidence-level');

    const period = ADOPTION_PERIODS[prediction.class];

    if (predictionTitle) {
        predictionTitle.textContent = `Expected Adoption: ${period.label}`;
    }

    if (predictionDescription) {
        predictionDescription.textContent = period.description;
    }

    if (confidenceLevel) {
        confidenceLevel.textContent = `${prediction.confidence}%`;
    }

    // Store prediction in petData
    petData.prediction = prediction.class;
    petData.confidence = prediction.confidence;
    petData.score = prediction.score;

    if (modal) {
        modal.classList.add('active');
    }
}

/**
 * Closes the prediction modal
 */
function closePredictionModal() {
    const modal = document.getElementById('prediction-modal');
    if (modal) {
        modal.classList.remove('active');
    }
}

/**
 * Saves the pet profile to local storage and updates the dashboard
 */
function savePetProfile() {
    if (petData.editingId) {
        // Update existing pet
        const index = savedPets.findIndex(p => p.id === petData.editingId);
        if (index !== -1) {
            // Preserve the original ID and date
            petData.id = petData.editingId;
            petData.dateAdded = savedPets[index].dateAdded;
            savedPets[index] = { ...petData };
            delete petData.editingId;

            // Save to localStorage
            localStorage.setItem('savedPets', JSON.stringify(savedPets));

            // Close modal
            closePredictionModal();

            // Show success message
            alert('Pet profile updated successfully!');

            // Return to dashboard
            showDashboard();

            return;
        }
    }

    // Add new pet (existing logic)
    savedPets.push(petData);

    // Save to localStorage
    localStorage.setItem('savedPets', JSON.stringify(savedPets));

    // Close modal
    closePredictionModal();

    // Show success message
    alert('Pet profile saved successfully!');

    // Return to dashboard
    showDashboard();
}

/**
 * Handles form submission, validates current step, and saves pet data
 * @param {Event} event - Form submission event
 * @return {void}
 */
function handleFormSubmit(event) {
    event.preventDefault();
    generatePrediction();
}

/**
 * Deletes a pet from local storage and backend with confirmation
 * @param {string} petId - Unique pet identifier to delete
 */
function deletePet(petId) {
    if (!petId) {
        alert('Pet ID is required for deletion');
        return;
    }

    // Find the pet in savedPets to show details in confirmation
    const pet = savedPets.find(p => p.id === petId);
    const petName = pet ? (pet.name || 'Unnamed Pet') : 'Unknown Pet';

    // Show confirmation dialog with pet details
    if (!confirm(`Are you sure you want to delete "${petName}"?\n\nThis action cannot be undone and will remove the pet from both the dashboard and CSV file.`)) {
        return;
    }

    // Delete from local storage
    const originalLength = savedPets.length;
    savedPets = savedPets.filter(pet => pet.id !== petId);

    if (savedPets.length === originalLength) {
        alert('Pet not found in local storage');
        return;
    }

    // Update local storage
    localStorage.setItem('savedPets', JSON.stringify(savedPets));

    // Delete from backend/CSV if available
    deletePetFromBackend(petId);

    // Update dashboard
    updateDashboard();

    // Show success message
    alert(`"${petName}" has been deleted successfully!`);
}

/**
 * Attempts to delete pet from backend server (optional)
 * @param {string} petId - Pet ID to delete from backend
 */
async function deletePetFromBackend(petId) {
    try {
        const response = await fetch('http://localhost:5000/delete', {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                pet_id: petId
            })
        });

        const result = await response.json();

        if (result.success) {
            console.log(`Pet ${petId} deleted from backend successfully`);
        } else {
            console.warn(`Failed to delete pet ${petId} from backend`);
        }
    } catch (error) {
        console.warn('Backend delete failed (backend may not be running):', error);
        // Don't show error to user since local deletion already worked
    }
}

/**
 * Shows detailed pet information in a modal dialog
 * @param {string} petId - Unique pet identifier
 */
function showPetDetails(petId) {
    console.log('showPetDetails called with petId:', petId); // Debug log

    const pet = savedPets.find(p => p.id === petId);

    if (!pet) {
        console.error('Pet not found:', petId);
        console.log('Available pets:', savedPets.map(p => p.id));
        return;
    }

    console.log('Pet found:', pet); // Debug log

    // Close any existing modal first
    closePetModal();

    const modal = document.createElement('div');
    modal.className = 'pet-modal-overlay';
    modal.onclick = (e) => {
        if (e.target === modal) {
            closePetModal();
        }
    };

    const breedName = getBreedName(pet.type, pet.breed);
    const stateName = getStateName(pet.state);
    const typeName = getTypeName(pet.type);
    const period = getPredictionPeriod(pet.prediction);

    modal.innerHTML = `
        <div class="pet-modal-content">
            <div class="pet-modal-header">
                <h2>${pet.name || 'Unnamed Pet'}</h2>
                <div class="modal-header-buttons">
                    <button class="modal-edit-btn" onclick="editPet('${pet.id}')">
                        <i class="fas fa-edit"></i> Edit
                    </button>
                    <button class="modal-close-btn" onclick="closePetModal()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
            <div class="pet-modal-body">
                <div class="pet-modal-image">
                    <img src="${pet.photos && pet.photos.length > 0 ? pet.photos[0] : 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><rect width="400" height="300" fill="%23f1f5f9"/><text x="200" y="150" text-anchor="middle" fill="%2364748b" font-family="Arial" font-size="16">No Image Available</text></svg>'}" 
                         alt="${pet.name || 'Pet'}" class="modal-pet-image">
                </div>
                <div class="pet-modal-info">
                    <div class="pet-modal-section">
                        <h3>Basic Information</h3>
                        <div class="pet-info-grid">
                            <div class="info-item">
                                <span class="info-label">Type:</span>
                                <span class="info-value">${typeName}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Breed:</span>
                                <span class="info-value">${breedName}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Age:</span>
                                <span class="info-value">${pet.age} months</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Gender:</span>
                                <span class="info-value">${getGenderName(pet.gender)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Color:</span>
                                <span class="info-value">${getColorName(pet.color)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Size:</span>
                                <span class="info-value">${getMaturitySizeName(pet.maturitySize)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Fur Length:</span>
                                <span class="info-value">${getFurLengthName(pet.furLength)}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="pet-modal-section">
                        <h3>Health & Care</h3>
                        <div class="pet-info-grid">
                            <div class="info-item">
                                <span class="info-label">Vaccinated:</span>
                                <span class="info-value">${getVaccinationStatus(pet.vaccinated)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Dewormed:</span>
                                <span class="info-value">${getVaccinationStatus(pet.dewormed)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Sterilized:</span>
                                <span class="info-value">${getVaccinationStatus(pet.sterilized)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Health:</span>
                                <span class="info-value">${getHealthStatus(pet.health)}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="pet-modal-section">
                        <h3>Adoption Details</h3>
                        <div class="pet-info-grid">
                            <div class="info-item">
                                <span class="info-label">Arrival Date:</span>
                                <span class="info-value">${formatDate(pet.dateAdded)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Predicted Adoption:</span>
                                <span class="info-value">${period.days}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Fee:</span>
                                <span class="info-value">$${parseFloat(pet.fee || 0).toFixed(0)}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Location:</span>
                                <span class="info-value">${stateName}</span>
                            </div>
                            <div class="info-item">
                                <span class="info-label">Confidence:</span>
                                <span class="info-value">${(pet.confidence || 0).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                    
                    ${pet.description ? `
                    <div class="pet-modal-section">
                        <h3>Description</h3>
                        <p class="pet-description">${pet.description}</p>
                    </div>
                    ` : ''}
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    document.body.style.overflow = 'hidden';
}

/**
 * Closes the pet details modal and restores page scrolling
 */
function closePetModal() {
    const modal = document.querySelector('.pet-modal-overlay');
    if (modal) {
        modal.remove();
        document.body.style.overflow = 'auto';
    }
}

// Edit pet functionality
function editPet(petId) {
    const pet = savedPets.find(p => p.id === petId);
    if (!pet) {
        alert('Pet not found');
        return;
    }

    // Close modal
    closePetModal();

    // Switch to add pet form
    showAddPetForm();

    // Pre-fill the form with existing data
    setTimeout(() => {
        fillFormWithPetData(pet);
    }, 100);
}

/**
 * Fills the add/edit pet form with existing pet data
 * @param {Object} pet - Pet data object to fill the form with
 * @return {void}
 */
function fillFormWithPetData(pet) {
    // Fill basic information
    document.getElementById('pet-type').value = pet.type;
    document.getElementById('pet-name').value = pet.name || '';
    document.getElementById('age').value = pet.age;
    document.getElementById('quantity').value = pet.quantity;
    document.getElementById('fee').value = pet.fee;

    // Trigger breed dropdown update
    handlePetTypeChange();

    setTimeout(() => {
        // Fill physical characteristics
        document.getElementById('breed').value = pet.breed;
        document.getElementById('gender').value = pet.gender;
        document.getElementById('color').value = pet.color;
        document.getElementById('maturity-size').value = pet.maturitySize;
        document.getElementById('fur-length').value = pet.furLength;

        // Fill health information
        document.getElementById('vaccinated').value = pet.vaccinated;
        document.getElementById('dewormed').value = pet.dewormed;
        document.getElementById('sterilized').value = pet.sterilized;
        document.getElementById('health').value = pet.health;

        // Fill location
        document.getElementById('state').value = pet.state;

        // Fill description
        document.getElementById('description').value = pet.description || '';

        // Set photos if available
        if (pet.photos && pet.photos.length > 0) {
            uploadedPhotos = pet.photos.map((dataUrl, index) => ({
                file: null,
                dataUrl: dataUrl,
                name: `photo_${index + 1}.jpg`
            }));
            updatePhotoPreview();
        }

        // Store the original pet ID for updating
        petData.editingId = pet.id;

        // Update the form title to indicate editing
        const formTitle = document.querySelector('.section-title');
        if (formTitle) {
            formTitle.textContent = `Edit Pet Profile - ${pet.name || 'Unnamed Pet'}`;
        }
    }, 200);
}

function getFurLengthName(furLengthId) {
    const furLengthMap = { 1: 'Short', 2: 'Medium', 3: 'Long' };
    return furLengthMap[furLengthId] || 'Unknown';
}

/**
 * Exports all saved pets to a CSV file for download
 */
function exportAllToCSV() {
    if (savedPets.length === 0) {
        alert('No pets to export. Add some pets first!');
        return;
    }

    // Create CSV header
    const csvData = [
        'PetID,Type,Name,Age,Breed,Gender,Color,MaturitySize,FurLength,Vaccinated,Dewormed,Sterilized,Health,Quantity,Fee,State,RescuerID,PhotoAmt,Description,Prediction,Confidence,DateAdded'
    ];

    // Add each pet's data
    savedPets.forEach(pet => {
        const row = [
            pet.id,
            pet.type,
            `"${pet.name || ''}"`,
            pet.age,
            pet.breed,
            pet.gender,
            pet.color,
            pet.maturitySize,
            pet.furLength,
            pet.vaccinated,
            pet.dewormed,
            pet.sterilized,
            pet.health,
            pet.quantity,
            pet.fee,
            pet.state,
            pet.rescuerID || 1,
            pet.photoAmt,
            `"${pet.description || ''}"`,
            pet.prediction,
            pet.confidence.toFixed(2),
            pet.dateAdded
        ].join(',');

        csvData.push(row);
    });

    // Create and download CSV file
    const csvContent = csvData.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');

    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `petty_ai_pets_${new Date().toISOString().split('T')[0]}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        alert(`Successfully exported ${savedPets.length} pets to CSV file!`);
    } else {
        alert('CSV export is not supported in this browser');
    }
}

/**
 * Formats a date string into a readable format
 * @param {string} dateString - ISO date string
 * @returns {string} Formatted date (e.g., "Jan 15, 2025")
 */
function formatDate(dateString) {
    if (!dateString) return 'Unknown';

    try {
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return 'Unknown';

        const options = {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        };
        return date.toLocaleDateString('en-US', options);
    } catch (error) {
        return 'Unknown';
    }
}

/**
 * Shows all pets in the dashboard (not just recent ones)
 */
function showAllPets() {
    const recentPetsGrid = document.getElementById('recent-pets-grid');

    // Clear existing content
    recentPetsGrid.innerHTML = '';

    if (savedPets.length === 0) {
        recentPetsGrid.innerHTML = `
            <div class="no-pets-message">
                <i class="fas fa-paw"></i>
                <p>No pets added yet</p>
                <button class="btn btn-primary" onclick="showAddPetForm()">Add Your First Pet</button>
            </div>
        `;
        return;
    }

    // Show all pets (newest first)
    const allPets = [...savedPets].reverse();

    allPets.forEach(pet => {
        const petCard = createPetCard(pet);
        recentPetsGrid.appendChild(petCard);
    });

    // Update the section header to show "All Pets" instead of "Recent Pets"
    const sectionHeader = document.querySelector('.recent-pets h2');
    if (sectionHeader) {
        sectionHeader.textContent = `All Pets (${savedPets.length})`;
    }

    // Change "View All" to "Show Recent"
    const viewAllLink = document.querySelector('.view-all-link');
    if (viewAllLink) {
        viewAllLink.textContent = 'Show Recent';
        viewAllLink.onclick = (e) => {
            e.preventDefault();
            showRecentPets();
        };
    }
}

/**
 * Resets the display to show only recent pets (last 6)
 */
function showRecentPets() {
    // Reset to showing only recent pets
    updateRecentPets();

    // Reset the section header
    const sectionHeader = document.querySelector('.recent-pets h2');
    if (sectionHeader) {
        sectionHeader.textContent = 'Recent Pets';
    }

    // Reset "View All" link
    const viewAllLink = document.querySelector('.view-all-link');
    if (viewAllLink) {
        viewAllLink.textContent = 'View All';
        viewAllLink.onclick = (e) => {
            e.preventDefault();
            showAllPets();
        };
    }
}

/**
 * Gets state name from state ID
 * @param {number} stateId - State ID
 * @returns {string} State name
 */
function getStateName(stateId) {
    return STATE_DATA[stateId] || 'Unknown';
}

/**
 * Gets prediction period information from prediction class
 * @param {number} predictionClass - Prediction class (0-3)
 * @returns {Object} Period object with days, badge, etc.
 */
function getPredictionPeriod(predictionClass) {
    return ADOPTION_PERIODS[predictionClass] || ADOPTION_PERIODS[2];
}