# Text Classification using TensorFlow Lite Plugin for Flutter

## Overview
This repository demonstrates how to implement text sentiment classification in a Flutter application using the `tflite_flutter` plugin. The app uses a pre-trained TensorFlow Lite model to predict whether a text paragraph has positive or negative sentiment.

## Key Features of `tflite_flutter`
- Provides a Dart API similar to the TensorFlow Lite Java and Swift APIs
- Directly binds to the TensorFlow Lite C API using dart:ffi for better efficiency
- No platform-specific code required
- Supports acceleration using NNAPI and GPU delegates on Android, and Metal delegate on iOS

## Getting Started

![DEMO GIF](demo.gif)

### Installation

1. **Initial setup (platform-specific)**:

   **Linux and Mac users**:
   ```bash
   sh install.sh
   ```

   **Windows users**:
   ```
   install.bat
   ```
   
   This will automatically download the latest binaries from release assets and place them in appropriate folders.

2. **Add the plugin to your pubspec.yaml**:
   ```yaml
   dependencies:
     tflite_flutter: ^<latest_version>
   ```

3. **Include model assets**:
   ```yaml
   assets:
     - assets/
   ```

4. **Add model files to assets folder**:
   - Download the [text classification model](https://www.tensorflow.org/lite/examples/text_classification/overview)
   - Place `text_classification.tflite` and `text_classification_vocab.txt` in your `assets/` directory

## Implementation

### The Classifier

Create a `classifier.dart` file with the following components:

1. **Load dictionary from vocabulary file**
2. **Tokenize input text**
3. **Create TensorFlow Lite interpreter**
4. **Perform inference on the tokenized input**

```dart
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class Classifier {
  // Classifier
  Interpreter _interpreter;
  
  // Load dictionary from assets
  Map<String, int> _dict;
  
  // Input shape => [1, 256]
  // Output shape => [1, 2]
  
  Classifier() {
    // Load model
    _loadModel();
    // Load dictionary
    _loadDictionary();
  }
  
  void _loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/text_classification.tflite');
    print('Interpreter loaded successfully');
  }
  
  void _loadDictionary() async {
    final vocab = await rootBundle.loadString('assets/text_classification_vocab.txt');
    var dict = <String, int>{};
    final vocabList = vocab.split('\n');
    for (var i = 0; i < vocabList.length; i++) {
      var entry = vocabList[i].trim();
      if (entry.isNotEmpty) {
        dict[entry] = i;
      }
    }
    _dict = dict;
    print('Dictionary loaded successfully');
  }
  
  List<int> tokenize(String text) {
    // Tokenize input text
    final toks = text.split(' ');
    var vec = List<int>.filled(256, 0);
    var i = 0;
    for (var tok in toks) {
      if (i >= 256) {
        break;
      }
      var id = _dict[tok.toLowerCase()];
      if (id != null) {
        vec[i++] = id;
      }
    }
    return vec;
  }
  
  int classify(String text) {
    // Input shape [1, 256]
    var input = List<List<int>>.filled(1, List<int>.filled(256, 0));
    input[0] = tokenize(text);
    
    // Output shape [1, 2]
    var output = List<List<double>>.filled(1, List<double>.filled(2, 0));
    
    // Run inference
    _interpreter.run(input, output);
    
    // Get results
    var result = output[0];
    
    // Return index with highest probability
    return result.indexOf(result.reduce((a, b) => a > b ? a : b));
  }
}
```

### Usage

```dart
// Create Classifier object
Classifier _classifier = Classifier();

// Call classify method with sentence as parameter
int result = _classifier.classify("I liked the movie");
// returns 1 (POSITIVE)

result = _classifier.classify("I didn't like the movie");
// returns 0 (NEGATIVE)
```

## FAQs

**Q: How is this plugin `tflite_flutter` different from `tflite` v1.0.5?**

While `tflite` v1.0.5 focuses on offering high-level features for specific use cases like Image Classification and Object Detection, `tflite_flutter` offers the same flexibility and features as the Java API and can be used with any TFLite model. It also provides support for delegates.

`tflite_flutter` is faster as it uses dart:ffi (dart ↔️(ffi) ↔️C) while `tflite` uses platform integration (dart ↔️platform-channel ↔️(Java/Swift) ↔️JNI ↔️C).

**Q: How to create an Image Classification app using `tflite_flutter`?**

TFLite Flutter Helper Library provides a simple architecture for processing and manipulating input and output of TFLite Models. Its API design and documentation are identical to the TensorFlow Lite Android Support Library.

## Resources

- [Complete Text Classification Example App](https://github.com/am15h/tflite_flutter_plugin/tree/master/example)
- [tflite_flutter_plugin on GitHub](https://github.com/am15h/tflite_flutter_plugin)
- [TFLite Flutter Helper Library](https://github.com/am15h/tflite_flutter_helper)


