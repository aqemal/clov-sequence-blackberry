"""Command-line read-only APK metadata inventory; does not run malware."""
import argparse
from ingestion import apk_inventory, inventory_summary, write_manifest

def main():
    p=argparse.ArgumentParser(description='Inventory APK metadata without running APKs')
    p.add_argument('folder', help='Local dataset folder (family/variety/file.apk)')
    p.add_argument('--out',default='apk_manifest.csv',help='Output manifest CSV')
    p.add_argument('--sha256',action='store_true',help='Read files to calculate SHA-256')
    args=p.parse_args()
    df=apk_inventory(args.folder,compute_sha256=args.sha256)
    write_manifest(df,args.out)
    print(inventory_summary(df))
    print(f'Manifest saved to {args.out}')
if __name__=='__main__':
    main()
