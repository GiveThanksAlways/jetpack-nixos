{
  description = "Jetson Orin AGX NixOS Configuration";

  inputs = {
    # Match the jetpack-nixos project's nixpkgs version
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    # Jetpack NixOS - provides Jetson hardware support
    # Uses upstream anduril repo, but you can point to your fork:
    # jetpack-nixos.url = "github:YOUR_USERNAME/jetpack-nixos/your-branch";
    jetpack-nixos.url = "github:anduril/jetpack-nixos/master";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack-nixos, ... }@inputs: {
    nixosConfigurations.jetson = nixpkgs.lib.nixosSystem {
      system = "aarch64-linux";
      
      # Pass inputs to modules so they can access flake inputs
      specialArgs = { inherit inputs; };
      
      modules = [
        # Jetpack NixOS module - MUST come before your config
        jetpack-nixos.nixosModules.default
        
        # Your system configuration
        ./configuration.nix
      ];
    };
  };
}
