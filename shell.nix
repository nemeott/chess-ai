{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    python313
    zlib
    cairo
  ];

  # ensure the dynamic linker env points at the nix store libs
  shellHook = ''
    # Compute the nix library path for the needed runtime closures
    export NIX_LD_LIBRARY_PATH=${
      with pkgs;
      lib.makeLibraryPath [
        zlib
        cairo
      ]
    }

    # Make the dynamic loader also see the nix path (this helps processes started in the shell)
    export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH''${LD_LIBRARY_PATH:+:}''$LD_LIBRARY_PATH"
  '';
}
