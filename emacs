;; major-mode for .fox files
(defvar foxy-constants
  '("AMU" "AMUMEV" "EZERO" "CLIGHT" "PI" "DEGRAD"
    "P0" "E0" "M0" "Z0" "V0" "CHIM" "CHIE" "ETA"))
  
(defvar foxy-keywords
  '("BEGIN" "END" "VARIABLE" "PROCEDURE" "ENDPROCEDURE" "FUNCTION" "ENDFUNCTION"
    "IF" "ELSEIF" "ENDIF" "WHILE" "ENDWHILE" "LOOP" "ENDLOOP" "PLOOP" "ENDPLOOP"
    "FIT" "ENDFIT" "WRITE" "READ" "SAVE" "INCLUDE"))

(defvar foxy-builtin-functions
  '("RE" "ST" "SF" "LO" "CM" "VE" "DA" "CD" "LRE" "LST" "LLO" "LCM" "LVE"
    "LDA" "LCD" "LGR" "TYPE" "LENGTH" "VARMEM" "VARPOI" "EXP" "LOG"
    "SIN" "COS" "TAN" "ASIN" "ACOS" "ATAN" "SINH" "COSH" "TANH" "SQRT"
    "ISRT" "ISRT3" "SQR" "ERF" "WERF" "VMIN" "VMAX" "ABS" "NORM" "CONS"
    "REAL" "IMAG" "CMPLX" "CONJ" "INT" "NINT" "NOT" "TRIM" "LTRIM" "GRIU"
    "OPENF" "CLOSEF" "OS" ))

(defvar foxy-tab-width 2 "Width of a tab for FOXY mode")

(defvar foxy-mode-syntax-table nil "Syntax table for `foxy-mode'.")

(defvar foxy-mode-map nil "Keymap for foxy-mode")
(progn
  (setq foxy-mode-map (make-sparse-keymap))
  (define-key foxy-mode-map (kbd "C-c C-<left>") 'indent-rigidly-left-to-tab-stop)
  (define-key foxy-mode-map (kbd "C-c C-<right>") 'indent-rigidly-right-to-tab-stop)
  )

(setq foxy-mode-syntax-table
      (let ( (synTable (make-syntax-table)))
        ;; comment style
        (modify-syntax-entry ?{ "<" synTable)
        (modify-syntax-entry ?} ">" synTable)
	;; single quotes also text
	(modify-syntax-entry ?' "\"" synTable)
        synTable))

(defvar foxy-font-lock-defaults
  `((
     (";\\|{\\|}\\|:=" . font-lock-comment-delimiter-face)
     ( ,(regexp-opt foxy-keywords 'words) . font-lock-function-name-face)
     ( ,(regexp-opt foxy-constants 'words) . font-lock-constant-face)
     ( ,(regexp-opt foxy-builtin-functions 'words) . font-lock-keyword-face)
     )))

(define-derived-mode foxy-mode prog-mode "FOXY language script"
  (setq font-lock-defaults foxy-font-lock-defaults)

  (setq tab-width 2)
  (setq indent-tabs-mode nil)
  (setq c-tab-always-indent t)

  (set-syntax-table foxy-mode-syntax-table)
  )
(provide 'foxy-mode)

;; this is so that comment-end syntax doesn't stick out in other modes
(add-hook 'foxy-mode-hook
          (lambda ()
            (set (make-local-variable 'comment-start) "{")
            (set (make-local-variable 'comment-end) "}")))

(add-to-list 'auto-mode-alist '("\\.fox\\'" . foxy-mode))
