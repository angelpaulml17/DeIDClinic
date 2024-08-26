// scripts.js
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('settingsForm');
    form.addEventListener('submit', function(event) {
        event.preventDefault();  // Prevent the form from submitting in the traditional way
        const action = document.querySelector('input[name="action"]:checked').value;
        const formData = new FormData(settingsForm);
        formData.append('action', action);

        fetch('/process_deidentification', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);  // Show a success message
            console.log('Success:', data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to save settings');
        });
    });
});
document.addEventListener('DOMContentLoaded', function() {
    const batchUploadForm = document.getElementById('batchUploadForm');
    const deidentifiedText = document.getElementById('deidentifiedText1');
    const originalText = document.getElementById('originalText1');
    const navigationButtons = document.getElementById('navigationButtons');
    const prevButton = document.getElementById('prevButton');
    const nextButton = document.getElementById('nextButton');
    let currentFileIndex = 0;
    let fileResults = [];

    batchUploadForm.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(batchUploadForm);
        fetch('/process_batch', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.text().then(text => { throw new Error(text) });
            }
            return response.json();
        })
        .then(data => {
            fileResults = data;
            console.log(fileResults);
            if (fileResults.action == 'redact'){
                currentFileIndex = 0;
            
            displayFile(currentFileIndex);
            
            navigationButtons.style.display = 'block';
            }
            else{
            currentFileIndex = 0;
            
            displayFile(currentFileIndex);
            
            navigationButtons.style.display = 'block';
            handleBatchResults(fileResults);}
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to process batch: ' + error.message);
        });
    });

    prevButton.addEventListener('click', function() {
        if (currentFileIndex > 0) {
            currentFileIndex--;
            displayFile(currentFileIndex);
        }
    });

    nextButton.addEventListener('click', function() {
        if (currentFileIndex < fileResults.results.length - 1) {
            currentFileIndex++;
            displayFile(currentFileIndex);
        }
    });

    function displayFile(index) {
        const result = fileResults.results[index];
        console.log(result);
        deidentifiedText.innerHTML = ''; // Clear previous content
        originalText.innerHTML = ''; // Clear previous content

        const titleOriginal = document.createElement('h3');
        titleOriginal.textContent = 'Original Clinical Letter';

        const contentOriginal = document.createElement('div');
        contentOriginal.innerHTML = result.original; // Use innerHTML to properly render tags

        const titleRedacted = document.createElement('h3');
        titleRedacted.textContent = 'Redacted Clinical Letter';

        const contentRedacted = document.createElement('div');
        contentRedacted.innerHTML = result.redacted; // Use innerHTML to properly render tags

        deidentifiedText.appendChild(titleOriginal);
        deidentifiedText.appendChild(contentOriginal);
        originalText.appendChild(titleRedacted);
        originalText.appendChild(contentRedacted);
        
    }


    document.getElementById('deidentifiedText1').addEventListener('mouseup', function() {});

    document.querySelector('button[onclick="removeSelectedEntity()"]').addEventListener('click', removeSelectedEntity);
    document.querySelector('button[onclick="markSelectedText()"]').addEventListener('click', markSelectedText);
    
});
function wrapSelectedTextWithSpan() {
    const selection = window.getSelection();
    if (selection.rangeCount === 0) return;
    const entityType = document.getElementById('entityType').value;
    const range = selection.getRangeAt(0);
    const selectedText = range.toString().trim();
    if (!selectedText) return;
    storedText = selectedText;
    // Create a span element with a unique ID
    const span = document.createElement('span');
    const uniqueId = `entity-${Date.now()}`;
    span.classList.add('highlighted-text');
    span.id = uniqueId;
    span.style.backgroundColor = colorMap[entityType]; // Optional: Add inline styles

    range.surroundContents(span);

    console.log(`Wrapped selected text: ${selectedText} with ID: ${uniqueId}`);

    return uniqueId;
}

function markSelectedText() {
    const uniqueId = wrapSelectedTextWithSpan();
    if (!uniqueId) return;

    const entityType = document.getElementById('entityType').value;
    const modal = document.getElementById('markEntityModal');
    modal.style.display = 'block';
    const action = document.querySelector('input[name="action"]:checked').value;
    if (action == "redact") {
        
        originalRHSContent = document.getElementById('originalText1').innerHTML;
        document.getElementById('markAllButton').onclick = function() {
            performMarking(true);
            modal.style.display = 'none';
        };
        document.querySelector('.close').onclick = function() {
            modal.style.display = 'none';
        };
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        };
    } else {
        
        originalRHSContent = document.getElementById('originalText1').innerHTML;
        document.getElementById('markAllButton').onclick = function() {
            sendTextToServer1(storedText, entityType);
            modal.style.display = 'none';
        };
        document.querySelector('.close').onclick = function() {
            modal.style.display = 'none';
        };
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        };
    }
}

function sendTextToServer1(text, entityType) {
    const sourceText = document.getElementById('deidentifiedText1').innerText;
    fetch('/update_and_deidentify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sourceText:sourceText, text: text, entity_type: entityType}),
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response data
        document.getElementById('originalText1').innerHTML = data.deidentifiedText;
        document.getElementById('deidentifiedText1').innerHTML = data.originalText;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function performMarkingOne(markAll, uniqueId, entityType) {
    console.log('performing marking one');
    const sourceDiv = document.getElementById('deidentifiedText1');
    const targetDiv = document.getElementById('originalText1');
    const span = document.getElementById(uniqueId);
    if (span) {
        const position = countTokensBeforeSpan(span.id);
        console.log(countTokensBeforeSpan(span.id));
        markTextAtPositionOne(sourceDiv, targetDiv, uniqueId, span.innerText, entityType, position);
    }
}

function countTokensBeforeSpan(spanId) {
    const span = document.getElementById(spanId);
    if (!span) {
        console.error(`Span with ID ${spanId} not found.`);
        return -1;
    }

    // Get the parent element containing the text
    const parent = span.parentNode;
    const fullText = parent.innerText;

    // Split the full text into tokens (words)
    const tokens = fullText.split(/\s+/);

    // Traverse the text content and count the tokens before the span
    let tokenCount = 0;
    let found = false;

    for (const token of tokens) {
        // Check if the current token is within the span
        const tempRange = document.createRange();
        tempRange.selectNodeContents(parent);
        const tempDiv = document.createElement('div');
        tempDiv.appendChild(tempRange.cloneContents());
        const tempSpan = tempDiv.querySelector(`#${spanId}`);
        if (tempSpan && tempSpan.innerText.includes(token)) {
            found = true;
            break;
        }
        tokenCount++;
    }

    return found ? tokenCount : -1;
}

function markTextAtPositionOne(sourceDiv, targetDiv, uniqueId, text, entityType, position) {
    const sourceText = sourceDiv.innerText.split(/\s+/);
    console.log('marking at position one');
    const targetText = targetDiv.innerText.split(/\s+/);
    console.log(position);
    if (position !== -1 && position < sourceText.length) {
        sourceText[position] = `<span class="${entityType}" style="background-color: ${colorMap[entityType]}" data-unique-id="${uniqueId}">${text}</span>`;
        targetText[position] = `XXX-${entityType.toUpperCase()}`;
    }

    sourceDiv.innerHTML = sourceText.join(' ');
    targetDiv.innerHTML = targetText.join(' ');
    recolorBasedOnEntityType(targetDiv);
}

function recolorBasedOnEntityType(targetDiv) {
    const text = targetDiv.innerHTML;
    const tokens = text.split(/\s+/);
    const newHTML = tokens.map(token => {
        const match = token.match(/XXX-([A-Z]+)/i);
        if (match && colorMap[match[1].toUpperCase()]) {
            return `<span class="${match[1]}" style="background-color: ${colorMap[match[1].toUpperCase()]}">${token}</span>`;
        }
        return token;
    }).join(' ');
    targetDiv.innerHTML = newHTML;
}

function performMarking(markAll) {
    const sourceDiv = document.getElementById('deidentifiedText1');
    const targetDiv = document.getElementById('originalText1');
    const uniqueId = `entity-${Date.now()}`;
    const entityType = document.getElementById('entityType').value;

    if (markAll) {
        const positions = getWordPositions(sourceDiv.innerText, storedText);
        console.log(positions);
        positions.forEach(pos => {
            markTextAtPosition(sourceDiv, targetDiv, storedText, entityType, pos, uniqueId);
        });
    } else {
        const position = getWordPosition(sourceDiv.innerText, storedText);
        console.log(position);
        markTextAtPosition(sourceDiv, targetDiv, storedText, entityType, position, uniqueId);
    }
}

function markTextAtPosition(sourceDiv, targetDiv, text, entityType, position, uniqueId) {
    const words = sourceDiv.innerText.split(/\s+/);
    const targetWords = targetDiv.innerText.split(/\s+/);

    if (position !== -1) {
        const span = document.createElement('span');
        span.classList.add(entityType);
        span.textContent = words[position];
        span.style.backgroundColor = colorMap[entityType];
        span.setAttribute('data-unique-id', uniqueId);

        words[position] = span.outerHTML;
        targetWords[position] = `XXX-${entityType.toUpperCase()}`;
    }

    sourceDiv.innerHTML = words.join(' ');
    targetDiv.innerHTML = targetWords.join(' ');
    recolorBasedOnEntityType(targetDiv);
}

function updateRedactedText(sourceDiv, targetDiv, selectedText, entityType) {
    const positions = getWordPositions(sourceDiv.innerText, selectedText);
    const targetWords = targetDiv.innerText.split(/\s+/);

    positions.forEach(pos => {
        if (pos !== -1) {
            targetWords[pos] = `XXX-${entityType.toUpperCase()}`;
        }
    });
    targetDiv.innerText = targetWords.join(' ');
}

function removeSelectedEntity() {
    const sourceDiv = document.getElementById('deidentifiedText1');
    const targetDiv = document.getElementById('originalText1');
    const selection = window.getSelection();
    if (selection.rangeCount === 0) return;

    const range = selection.getRangeAt(0);
    const selectedText = range.toString();
    if (!selectedText.trim()) return;

    storedRange = range;
    storedText = selectedText;
    storedEntityType = document.getElementById('entityType').value;

    const modal = document.getElementById('replaceModal');
    modal.style.display = 'block';

    document.getElementById('replaceOne').onclick = function() {
        replaceText(false);
        modal.style.display = 'none';
    };

    document.getElementById('replaceAll').onclick = function() {
        replaceText(true);
        modal.style.display = 'none';
    };

    document.querySelector('.close').onclick = function() {
        modal.style.display = 'none';
    };

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    };
}

function replaceText(replaceAll) {
    const sourceDiv = document.getElementById('deidentifiedText1');
    const targetDiv = document.getElementById('originalText1');
    const targetWords = targetDiv.innerText.split(/\s+/);
    const action = document.querySelector('input[name="action"]:checked').value;
    const entityType = document.getElementById('entityType').value;

    if (action === 'redact') {
        if (replaceAll) {
            const positions = getWordPositions(sourceDiv.innerText, storedText);
            positions.forEach(pos => {
                if (pos !== -1 && targetWords[pos].startsWith('XXX-')) {
                    targetWords[pos] = storedText;
                }
            });
        } else {
            const position = getWordPosition(sourceDiv.innerText, storedText);
            if (position !== -1 && targetWords[position].startsWith('XXX-')) {
                targetWords[position] = storedText;
            }
        }
        targetDiv.innerHTML = targetWords.join(' ');
        recolorBasedOnEntityType(targetDiv);
    } else {
        const childInfoList = Array.from(targetDiv.children[1].children).map(child => ({
            textContent: child.childNodes[0] ? child.childNodes[0].data : null,
            firstAttribute: child.attributes[0] ? child.attributes[0].nodeValue : null
        }));
        if (replaceAll) {
            const positions = getWordPositions(sourceDiv.innerText, storedText);
            positions.forEach(pos => {
                if (pos !== -1) {
                    targetWords[pos] = `<span class='replaced ${entityType}' style='background-color: ${colorMap[entityType]}'>${storedText}</span>`;
                }
            });
            targetDiv.innerHTML = highlightText(targetWords.join(' '), childInfoList);
        } else {
            const position = getWordPosition(sourceDiv.innerText, storedText);
            if (position !== -1) {
                targetWords[position] = `<span class='replaced ${entityType}' style='background-color: ${colorMap[entityType]}'>${storedText}</span>`;
            }
            targetDiv.innerHTML = highlightText(targetWords.join(' '), childInfoList);
        }
    }

    const spans = sourceDiv.getElementsByTagName('span');
    for (let span of spans) {
        if (span.textContent === storedText) {
            span.outerHTML = storedText;
        }
    }

    storedRange = null;
    storedText = null;
    storedEntityType = null;
}

function highlightText(targetDivMark, childInfoList) {
    for (let childInfo of childInfoList) {
        if (!childInfo.textContent) continue;
        let regex = new RegExp(`(${childInfo.textContent})`, 'g');
        targetDivMark = targetDivMark.replace(regex, (match) => {
            return `<span style='${childInfo.firstAttribute}';>${match}</span>`;
        });
    }
    return targetDivMark;
}

function getWordPositions(text, word) {
    const words = text.split(/\s+/);
    const positions = [];
    const cleanedWord = removePunctuation(word);
    for (let i = 0; i < words.length; i++) {
        if (removePunctuation(words[i]) === cleanedWord) {
            positions.push(i);
        }
    }
    return positions;
}

function getWordPosition(text, word) {
    const words = text.split(/\s+/);
    const cleanedWord = removePunctuation(word);
    for (let i = 0; i < words.length; i++) {
        if (removePunctuation(words[i]) === cleanedWord) {
            return i;
        }
    }
    return -1;
}

function removePunctuation(text) {
    return text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
}


// Define a color map for different entity types
colorMap = {
    'Name': '#ffc5d9',
    'Age': '#c2f2d0',
    'Date': '#ffcb85',
    'Profession': 'brown',
    'Location': 'green'
}

document.addEventListener('DOMContentLoaded', function() {
    colorMap = {
        'Name': '#ffc5d9',
        'Age': '#c2f2d0',
        'Date': '#ffcb85',
        'Profession': 'brown',
        'Location': 'green'
    };

    let storedRange, storedText, storedEntityType, storedStartOffset, storedEndOffset, originalRHSContent;


    function updateRedactedText(sourceDiv, targetDiv, uniqueId, entityType) {
        // Clone the HTML of the target div to manipulate it
        let targetHTML = targetDiv.innerHTML;

        // Find the newly marked span in the source div using the unique ID
        let markedSpan = sourceDiv.querySelector(`span[data-unique-id='${uniqueId}']`);

        if (markedSpan) {
            let markedText = markedSpan.textContent;
            let regex = new RegExp(`\\b${escapeRegExp(markedText)}\\b`);

            // Replace only the specific marked text in the target div
            targetHTML = targetHTML.replace(regex, `<span class='redacted ${entityType}' style='background-color: ${colorMap[entityType]}'>XXX-${entityType.toUpperCase()}</span>`);
        }

        // Update the target div's HTML
        targetDiv.innerHTML = targetHTML;
    }

    // Helper function to escape regex special characters in text
    function escapeRegExp(text) {
        return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, '\\$&');
    }

    // Function to handle text selection
    

    // Function to check if text is already highlighted
    function isAlreadyHighlighted(text) {
        return Object.values(highlightedEntities).includes(text);
    }
    colorMapc = {
        'name': '#ffc5d9',
        'age': '#c2f2d0',
        'date': '#ffcb85',
        'profession': 'brown',
        'location': 'green'
    };
    // Function to remove highlight from the selected entity
    function removeSelectedEntity() {
        const sourceDiv = document.getElementById('deidentifiedText');
        const targetDiv = document.getElementById('originalText');
        const removeEntityType = document.getElementById('entityType').value;
        const selection = window.getSelection();
        if (selection.rangeCount === 0) return; // Exit if no text is selected

        const range = selection.getRangeAt(0);
        const selectedText = range.toString();
        if (!selectedText.trim()) return; // Exit if selected text is empty

        const entityType = document.getElementById('entityType').value;
        storedRange = range;
        storedEntityType = entityType;
        storedText = selectedText;
        storedStartOffset = range.startOffset;
        storedEndOffset = range.endOffset;

        // Show the modal dialog for replace options
        const modal = document.getElementById('replaceModal');
        modal.style.display = 'block';

        document.getElementById('replaceOne').onclick = function() {
            replaceText(false);
            modal.style.display = 'none';
        };

        document.getElementById('replaceAll').onclick = function() {
            replaceText(true);
            modal.style.display = 'none';
        };

        document.querySelector('.close').onclick = function() {
            modal.style.display = 'none';
        };

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        };
    }

    
    document.getElementById('deidentifiedText').addEventListener('mouseup', function() {
        
    });

    document.querySelector('button[onclick="removeSelectedEntity()"]').addEventListener('click', removeSelectedEntity);
    
});











document.addEventListener('DOMContentLoaded', function() {
    colorMap = {
    'name': '#ffc5d9',
    'age': '#c2f2d0',
    'date': '#ffcb85',
    'profession': 'brown',
    'location': 'green'
};

    let storedRange, storedText, storedEntityType, originalRHSContent;
    function wrapSelectedTextWithSpan() {
        const selection = window.getSelection();
        if (selection.rangeCount === 0) return;
        const entityType = document.getElementById('entityType').value;
        const range = selection.getRangeAt(0);
        const selectedText = range.toString().trim();
        if (!selectedText) return;
        storedText=selectedText;
        // Create a span element with a unique ID
        const span = document.createElement('span');
        const uniqueId = `entity-${Date.now()}`;
        span.classList.add('highlighted-text');
        span.id = uniqueId;
        span.style.backgroundColor = colorMap[entityType]; // Optional: Add inline styles
    
        range.surroundContents(span);
    
        console.log(`Wrapped selected text: ${selectedText} with ID: ${uniqueId}`);
    
        return uniqueId;
    }
    

    function markSelectedText() {
        const uniqueId = wrapSelectedTextWithSpan();
    if (!uniqueId) return;
    const sourceText = document.getElementById('deidentifiedText').innerText;
    const entityType = document.getElementById('entityType').value;
    const modal = document.getElementById('markEntityModal');
    modal.style.display = 'block';
    const action = document.querySelector('input[name="action"]:checked').value;
    if(action=="redact"){
    
    originalRHSContent = document.getElementById('originalText').innerHTML;
        document.getElementById('markAllButton').onclick = function() {
            performMarking(true);
            modal.style.display = 'none';
        };

        document.querySelector('.close').onclick = function() {
            modal.style.display = 'none';
        };

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        };}
        else{
            
            originalRHSContent = document.getElementById('originalText').innerHTML;
                document.getElementById('markAllButton').onclick = function() {
                    sendTextToServer(sourceText, storedText, entityType);
                    modal.style.display = 'none';
                };
        
                document.querySelector('.close').onclick = function() {
                    modal.style.display = 'none';
                };
        
                window.onclick = function(event) {
                    if (event.target == modal) {
                        modal.style.display = 'none';
                    }
                };
        }
    }
    
    function sendTextToServer(sourceText, text, entityType) {
        fetch('/update_and_deidentify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sourceText:sourceText, text: text, entity_type: entityType }),
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response data
            document.getElementById('originalText').innerHTML = data.deidentifiedText ;
            document.getElementById('deidentifiedText').innerHTML = data.originalText;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    function performMarkingOne(markAll, uniqueId, entityType) {
        console.log('perpos');
        const sourceDiv = document.getElementById('deidentifiedText');
        const targetDiv = document.getElementById('originalText');
    
        
            const span = document.getElementById(uniqueId);
            if (span) {
                const position = countTokensBeforeSpan(span.id);
                console.log(countTokensBeforeSpan(span.id));
                markTextAtPositionOne(sourceDiv, targetDiv, uniqueId, span.innerText, entityType, position);
            }
    }
    function isWordSurroundedBySpan(text, container, spanId) {
        let spans = container.getElementsByTagName('span');
        for (let span of spans) {
            if (span.innerText.trim() === text.trim() && span.id === spanId) {
                return span;
            }
        }
        return null;
    }
    function countTokensBeforeSpan(spanId) {
        const span = document.getElementById(spanId);
        if (!span) {
            console.error(`Span with ID ${spanId} not found.`);
            return -1;
        }

        // Get the parent element containing the text
        const parent = span.parentNode;
        const fullText = parent.innerText;

        // Split the full text into tokens (words)
        const tokens = fullText.split(/\s+/);

        // Traverse the text content and count the tokens before the span
        let tokenCount = 0;
        let found = false;

        for (const token of tokens) {
            // Check if the current token is within the span
            const tempRange = document.createRange();
            tempRange.selectNodeContents(parent);
            const tempDiv = document.createElement('div');
            tempDiv.appendChild(tempRange.cloneContents());
            const tempSpan = tempDiv.querySelector(`#${spanId}`);
            if (tempSpan && tempSpan.innerText.includes(token)) {
                found = true;
                break;
            }
            tokenCount++;
        }

        return found ? tokenCount : -1;
    }
    function getWordPositionMark(text, word, span, container) {
        console.log('getpos');
        const words = text.split(/\s+/);
        let position = -1;
        words.forEach((token, index) => {
            if (token === word && isWordSurroundedBySpan(token, container, span.id)) {
                position = index;
                return position;
            }
        });
        
    }
    
    function markTextAtPositionOne(sourceDiv, targetDiv, uniqueId, text, entityType, position) {
        const sourceText = sourceDiv.innerText.split(/\s+/);
        console.log('markpos');
        const targetText = targetDiv.innerText.split(/\s+/);
        console.log(position);
        if (position !== -1 && position < sourceText.length) {
            sourceText[position] = `<span class="${entityType}" style="background-color: ${colorMap[entityType]}" data-unique-id="${uniqueId}">${text}</span>`;
            console.log(entityType);
            targetText[position] = `XXX-${entityType.toUpperCase()}`;
        }
    
        targetDiv.innerHTML = targetText.join(' ');
        recolorBasedOnEntityType(targetDiv);
    }
    const colorMapr = {
        'NAME': '#ffc5d9',
        'AGE': '#c2f2d0',
        'DATE': '#ffcb85',
        'PROFESSION': 'brown',
        'LOCATION': 'green'
    };

    function recolorBasedOnEntityType(targetDiv) {
        const text = targetDiv.innerHTML;
        const tokens = text.split(/\s+/);
        const newHTML = tokens.map(token => {
            const match = token.match(/XXX-([A-Z]+)/i);
        if (match && colorMapr[match[1].toUpperCase()]) {
            return `<span class="${match[1]}" style="background-color: ${colorMapr[match[1].toUpperCase()]}">${token}</span>`;
        }
            return token;
        }).join(' ');
        targetDiv.innerHTML = newHTML;
    }

    function performMarking(markAll) {
        const sourceDiv = document.getElementById('deidentifiedText');
        const targetDiv = document.getElementById('originalText');
        const uniqueId = `entity-${Date.now()}`;
        const entityType = document.getElementById('entityType').value;

        if (markAll) {
            const positions = getWordPositions(sourceDiv.innerText, storedText);
            console.log(positions);
            console.log("all");
            positions.forEach(pos => {
                markTextAtPosition(sourceDiv, targetDiv, storedText, entityType, pos, uniqueId);
            });
        } else {
            const position = getWordPosition(sourceDiv.innerText, storedText);
            console.log(position);
            console.log("single");
            markTextAtPosition(sourceDiv, targetDiv, storedText, entityType, position, uniqueId);
        }
    }

    function markTextAtPosition(sourceDiv, targetDiv, text, entityType, position, uniqueId) {
        const words = sourceDiv.innerText.split(/\s+/);
        const targetWords = targetDiv.innerText.split(/\s+/);

        if (position !== -1) {
            const span = document.createElement('span');
            span.classList.add(entityType);
            span.textContent = words[position];
            span.style.backgroundColor = colorMap[entityType];
            span.setAttribute('data-unique-id', uniqueId);

            words[position] = span.outerHTML;
            targetWords[position] = `XXX-${entityType.toUpperCase()}`;
        }

        targetDiv.innerHTML = targetWords.join(' ');
        recolorBasedOnEntityType(targetDiv);
    }
    document.querySelector('button[onclick="markSelectedText()"]').addEventListener('click', markSelectedText);
    function updateRedactedText(sourceDiv, targetDiv, selectedText, entityType) {
        // Get the positions of the selected text in the source div
        const positions = getWordPositions(sourceDiv.innerText, selectedText);
        const targetWords = targetDiv.innerText.split(/\s+/);

        // Replace the corresponding words in the target div
        positions.forEach(pos => {
            if (pos !== -1) {
                targetWords[pos] = `XXX-${entityType.toUpperCase()}`;
            }
        });
        targetDiv.innerText = targetWords.join(' ');
    }

    function removeSelectedEntity() {
        const sourceDiv = document.getElementById('deidentifiedText');
        const targetDiv = document.getElementById('originalText');
        const selection = window.getSelection();
        const entityType = document.getElementById('entityType').value;
        if (selection.rangeCount === 0) return; // Exit if no text is selected

        const range = selection.getRangeAt(0);
        const selectedText = range.toString();
        if (!selectedText.trim()) return; // Exit if selected text is empty

        storedRange = range;
        storedText = selectedText;
        storedEntityType = document.getElementById('entityType').value;

        // Show the modal dialog for replace options
        const modal = document.getElementById('replaceModal');
        modal.style.display = 'block';

        document.getElementById('replaceOne').onclick = function() {
            replaceText(false);
            modal.style.display = 'none';
        };

        document.getElementById('replaceAll').onclick = function() {
            replaceText(true);
            modal.style.display = 'none';
        };

        document.querySelector('.close').onclick = function() {
            modal.style.display = 'none';
        };

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        };
    }

    function replaceText(replaceAll) {
        const sourceDiv = document.getElementById('deidentifiedText');
        const targetDiv = document.getElementById('originalText');
        console.log(targetDiv.children);
        console.log(targetDiv.children.Array);
        console.log(targetDiv.children[0].childNodes[0].data);
        console.log(targetDiv.children[0].attributes[0].nodeValue);
        const childInfoList = [];

    for (let child of targetDiv.children) {
        const childInfo = {
            textContent: child.childNodes[0] ? child.childNodes[0].data : null,
            firstAttribute: child.attributes[0] ? child.attributes[0].nodeValue : null
        };
        childInfoList.push(childInfo);
    }
    console.log(childInfoList);
        const targetWords = targetDiv.innerText.split(/\s+/);
        const action = document.querySelector('input[name="action"]:checked').value;
        const entityType = document.getElementById('entityType').value;
        if(action=='redact'){
        if (replaceAll) {
            const positions = getWordPositions(sourceDiv.innerText, storedText);
            console.log(positions)
            positions.forEach(pos => {
                if (pos !== -1 && targetWords[pos].startsWith('XXX-')) {
                    targetWords[pos] = storedText;
                }
            });
            
            targetDiv.innerHTML = targetWords.join(' ');
            recolorBasedOnEntityType(targetDiv);
        } else {
            const position = getWordPosition(sourceDiv.innerText, storedText);
            console.log(position)
            if (position !== -1 && targetWords[position].startsWith('XXX-')) {
                targetWords[position] = storedText;
            }
            
            targetDiv.innerHTML = targetWords.join(' ');
            recolorBasedOnEntityType(targetDiv);
        }
    }
    else{
        if (replaceAll) {
            const positions = getWordPositions(sourceDiv.innerText, storedText);
            console.log(positions)
            
            positions.forEach(pos => {
                if (pos !== -1) {
                    targetWords[pos] = `<span class='replaced ${entityType}' style='background-color: ${colorMapc[entityType]}'>${storedText}</span>`;
                }
                targetDiv.innerHTML=highlightText(targetWords.join(' '), childInfoList);
            });
        } else {
            const position = getWordPosition(sourceDiv.innerText, storedText);
            console.log(position)
            if (position !== -1) {
                targetWords[position] = `<span class='replaced ${entityType}' style='background-color: ${colorMapc[entityType]}'>${storedText}</span>`;
            }
            
            targetDiv.innerHTML=highlightText(targetWords.join(' '), childInfoList);
        }
    }

        

        // Remove the highlighting from the source div
        const spans = sourceDiv.getElementsByTagName('span');
        for (let span of spans) {
            if (span.textContent === storedText) {
                span.outerHTML = storedText; // Replace span with plain text
            }
        }

        // Clear the stored data
        storedRange = null;
        storedText = null;
        storedEntityType = null;
    }
    function highlightText(targetDivMark, childInfoList) {
        for (let childInfo of childInfoList) {
            if (!childInfo.textContent) continue;
    
            let regex = new RegExp(`(${childInfo.textContent})`, 'g');
            targetDivMark = targetDivMark.replace(regex, (match) => {
                return `<span style='${childInfo.firstAttribute}';>${match}</span>`;
            });
        }
        return targetDivMark;
    }
    function getWordPositions(text, word) {
        const words = text.split(/\s+/);
        const positions = [];
        const cleanedWord = removePunctuation(word);
        for (let i = 0; i < words.length; i++) {
            if (removePunctuation(words[i]) === cleanedWord) {
                positions.push(i);
            }
        }
        return positions;
    }

    function getWordPosition(text, word) {
        const words = text.split(/\s+/);
        const cleanedWord = removePunctuation(word);
        for (let i = 0; i < words.length; i++) {
            if (removePunctuation(words[i]) === cleanedWord) {
                return i;
            }
        }
        return -1;
    }

    function removePunctuation(text) {
        console.log(text);
        return text.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
    }

    document.getElementById('deidentifiedText').addEventListener('mouseup', function() {});

    document.querySelector('button[onclick="removeSelectedEntity()"]').addEventListener('click', removeSelectedEntity);
    document.querySelector('button[onclick="markSelectedText()"]').addEventListener('click', markSelectedText);
});




document.getElementById('deidentifiedText').addEventListener('input', function() {
    const originalText = document.getElementById('deidentifiedText');
    const redactedTextDiv = document.getElementById('originalText');
    updateRedactedText(originalText, redactedTextDiv);
});

document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/deidentify', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('originalText').innerHTML = data.originalText;
            document.getElementById('deidentifiedText').innerHTML = data.deidentifiedText;
        })
        .catch(error => console.error('Error:', error));
    }
});


document.getElementById('downloadButton').addEventListener('click', function() {
    const text = document.getElementById('originalText').innerText;
    const blob = new Blob([text], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'deidentified.txt';
    link.click();
});





document.getElementById('saveReviewButton').addEventListener('click', function() {
    const reviewContent = document.getElementById('reviewContent').innerText;
    const blob = new Blob([reviewContent], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'reviewed_deidentified.txt';
    link.click();
});

document.getElementById('batchProcessButton').addEventListener('click', function() {
    const batchInput = document.getElementById('batchInput');
    const files = batchInput.files;
    const results = [];
    Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const content = e.target.result;
            const deidentifiedContent = deidentifyText(content);
            results.push(deidentifiedContent);
            if (results.length === files.length) {
                displayBatchResults(results);
            }
        };
        reader.readAsText(file);
    });
});
// Function to calculate severity and return the corresponding color
function calculateSeverity(count, totalCount) {
    const percentage = (count / totalCount) * 100;
    if (percentage < 25) {
        return 'green';
    } else if (percentage >= 25 && percentage <= 50) {
        return 'yellow';
    } else {
        return 'red';
    }
}

// Function to display the high-risk files
function displayHighRiskFiles(highRiskFiles, totalCount) {
    const highRiskContainer = document.getElementById('highRiskFilesContainer');
    highRiskContainer.innerHTML = ''; // Clear previous content

    highRiskFiles.forEach(([filename, count]) => {
        const color = calculateSeverity(count, totalCount);
        const fileElement = document.createElement('div');
        fileElement.style.backgroundColor = color;
        fileElement.className = 'high-risk-file';
        fileElement.innerHTML = `
            <strong>File:</strong> ${filename} <br>
            <strong>Count:</strong> ${count} <br>
            <strong>Risk Level:</strong> ${color.toUpperCase()}
        `;
        highRiskContainer.appendChild(fileElement);
    });
}

// Example of how to use the displayHighRiskFiles function with results from the server
function handleBatchResults(batchResults) {
    const highRiskFiles = batchResults.filesWithHighRisk;
    const totalCount = highRiskFiles.reduce((acc, [filename, count]) => acc + count, 0);
    displayHighRiskFiles(highRiskFiles, totalCount);
}

function deidentifyText(text) {
    const tool = localStorage.getItem('deidentificationTool');
    const entities = JSON.parse(localStorage.getItem('deidentificationEntities'));
    
    // Implement deidentification logic based on the selected tool and entities
    // This is a simplified example, actual implementation may vary
    if (tool === 'mask') {
        entities.forEach(entity => {
            const regex = new RegExp(entity, 'gi');
            text = text.replace(regex, '[MASKED]');
        });
    } else if (tool === 'phileas') {
        // Implement Phileas tool deidentification logic
    } else if (tool === 'anoncat') {
        // Implement AnonCAT tool deidentification logic
    }

    return text;
}

function displayBatchResults(results) {
    const batchResults = document.getElementById('batchResults');
    batchResults.innerHTML = '';
    results.forEach((result, index) => {
        const resultItem = document.createElement('p');
        resultItem.innerText = `File ${index + 1}: ${result}`;
        batchResults.appendChild(resultItem);
    });
}

function highlightEntities() {
    const reviewContent = document.getElementById('reviewContent');
    let content = reviewContent.innerHTML;

    const entities = [
        { name: 'Patient Names', color: '#daf8e3' },
        { name: 'Dates', color: '#97ebdb' },
        { name: 'Medical IDs', color: 'lightgreen' },
        { name: 'Addresses', color: 'lightcoral' },
        { name: 'Contact Numbers', color: 'lightpink' }
    ];

    entities.forEach(entity => {
        const regex = new RegExp(`(${entity.name})`, 'gi');
        content = content.replace(regex, `<span style="background-color: ${entity.color};">$1</span>`);
    });

    reviewContent.innerHTML = content;
}
