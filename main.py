import argparse
from encoder import encoder
from decoder import decoder
import sys

encoder_obj = encoder(8, 256)
decoder_obj = decoder(8, 256)

parser = argparse.ArgumentParser(description='JPEG Steganography Tool, designed for use with social media', formatter_class=argparse.RawDescriptionHelpFormatter, epilog='Third Year Dissertation Project\n\nStudent: David Byfield\nEmail:david.byfield@durham.ac.uk\n\nSupervisor: Andrei Krokhin\nEmail: andrei.krokhin@durham.ac.uk')
group = parser.add_mutually_exclusive_group()
group.add_argument('-embed', action='store_true', help='Embed [message] in [img]. Will output jpeg/txt if verbose enabled/disabled')
group.add_argument('-extract', action='store_true', help='Extract [message] from [img]. Will output text file containing message')
parser.add_argument('--image', '-i', required=True, type=str, help='Path to input image to embed/extract message in')
parser.add_argument('--message', '-m', type=str, default=None, required='-embed' in sys.argv, help='Path to file containing message')
parser.add_argument('--path', '-p', type=str, default=None, required='-extract' in sys.argv, help='Path to file containing path of embedding for message')
parser.add_argument('--key', '-k', type=str, default=None, required='-extract' in sys.argv, help='Key to decrypt path file')
parser.add_argument('--output', '-o', type=str, default='stego', help='What the output image/message file will be called. Default: stego.jpg/.txt')
parser.add_argument('--rs', action='store_true', default=False, help='Enable Reed-Solomon encoding of message for more robust communication. NOTE: This must be used on extraction if it was used in embedding!')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='When enabled, program will produce text file of bitstring rather than a JPG and perform every step of the encoding/decoding process manually')
parser.add_argument('--function', '-f', type=int, choices=[0,1,2,3], default=0, help='The algorithm used for embedding. 0: F5 1: F5 w/ SDCS 2: F5 w/ DMCSS')

args = parser.parse_args()

if args.embed:
    encoder_obj.encode(args.image, args.message, args.function, args.verbose, args.rs, args.output)
elif args.extract:
    decoder_obj.decode(args.image, bytes(args.key, "utf8"), args.function, args.verbose, args.rs, args.output)