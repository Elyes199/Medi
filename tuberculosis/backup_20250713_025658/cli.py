import os
import argparse
from tuberculosis.tb_detect import process_image

def main():
    parser = argparse.ArgumentParser(description="Detect tuberculosis in chest X-ray images")
    parser.add_argument("--image", "-i", type=str, help="Path to the chest X-ray image")
    parser.add_argument("--batch", "-b", type=str, help="Process all images in the specified directory")
    args = parser.parse_args()

    if args.image:
        # Process a single image
        if not os.path.exists(args.image):
            print(f"Error: Image {args.image} does not exist")
            return
        
        print(f"Processing image: {args.image}")
        result = process_image(args.image)
        
        # Display results
        if result["detected"]:
            print(f"\n⚠️ TB DETECTED - Severity: {result['severity']}")
            print(f"Affected lung area: {result['lesion_percentage']:.2f}%")
        else:
            print("\n✓ No TB detected")
            
    elif args.batch:
        # Process all images in a directory
        if not os.path.exists(args.batch) or not os.path.isdir(args.batch):
            print(f"Error: Directory {args.batch} does not exist")
            return
        
        # List all image files
        image_files = [os.path.join(args.batch, f) for f in os.listdir(args.batch) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in {args.batch}")
            return
        
        print(f"Processing {len(image_files)} images from {args.batch}")
        
        # Process each image
        results = []
        for img_path in image_files:
            print(f"Processing {os.path.basename(img_path)}...")
            try:
                result = process_image(img_path)
                results.append({
                    "filename": os.path.basename(img_path),
                    "detected": result["detected"],
                    "severity": result["severity"],
                    "lesion_percentage": result["lesion_percentage"]
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        # Display summary
        tb_detected = sum(1 for r in results if r["detected"])
        print(f"\nSummary: TB detected in {tb_detected} out of {len(results)} images ({tb_detected/len(results)*100:.1f}%)")
        
        # Display detailed results
        print("\nDetailed results:")
        print("-" * 70)
        print(f"{'Filename':<30} {'TB Detected':<15} {'Severity':<15} {'Affected Area'}")
        print("-" * 70)
        
        for r in results:
            detected = "✓" if r["detected"] else "-"
            print(f"{r['filename']:<30} {detected:<15} {r['severity']:<15} {r['lesion_percentage']:.2f}%")
    
    else:
        # No arguments provided, show usage
        parser.print_help()
        print("\nExamples:")
        print("  Process a single image:")
        print("  python -m tuberculosis.cli --image tb_data/TestImages/sample.png")
        print("\n  Process all images in a directory:")
        print("  python -m tuberculosis.cli --batch tb_data/TestImages")


if __name__ == "__main__":
    main()
