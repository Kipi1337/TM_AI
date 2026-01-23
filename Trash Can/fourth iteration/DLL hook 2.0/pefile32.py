import pefile

def read_function_signature(dll_path, offset):
    # Load the PE file
    pe = pefile.PE(dll_path)

    # Read the bytes at the specified offset
    signature_bytes = pe.get_memory_mapped_image()[offset:offset+20]  # Adjust the length as needed

    # Convert bytes to a readable string
    signature_str = ''.join([chr(b) for b in signature_bytes if b >= 32 and b <= 126])

    return signature_bytes, signature_str

# Path to your DLL
dll_path = 'path/to/your/TMInterface.dll'

# Offset where get_PlayerInfo() is located
offset = 0x0028C950

# Read the function signature
bytes_signature, str_signature = read_function_signature(dll_path, offset)

print(f"Bytes Signature: {bytes_signature}")
print(f"String Signature: {str_signature}")
