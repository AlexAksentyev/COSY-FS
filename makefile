CORE-DIR= $(HOME)/REPOS/COSYINF-CORE
CORE = $(addsuffix .bin, cosy utilities elements)
SETUPS = $(addsuffix .bin, BNL EB BNL8gsd)
SUPPORT = $(addsuffix decoh_optim.bin, EB/ BNL/)

define RM
	find $(1) -type f -name $(2) -print -delete
endef

core.bld: $(addprefix $(CORE-DIR)/bin/, $(CORE))
	echo $(CORE) >> core.bld

setups.bld: $(addprefix bin/setups/, $(SETUPS))
	echo $(SETUPS) >> setups.bld

support.bld: $(addprefix bin/support/, $(SUPPORT))
	echo $(SUPPORT) >> support.bld

$(CORE-DIR)/bin/cosy.bin: $(CORE-DIR)/src/cosy.fox ;
	cosy $<;
$(CORE-DIR)/bin/utilities.bin: $(CORE-DIR)/src/utilities.fox $(CORE-DIR)/bin/cosy.bin 
	cosy $<;
$(CORE-DIR)/bin/elements.bin: $(CORE-DIR)/src/elements.fox $(CORE-DIR)/bin/utilities.bin 
	cosy $<;

bin/elements.bin: src/elements.fox $(CORE-DIR)/bin/utilities.bin
	cosy $<;
bin/setups/%.bin: src/setups/%.fox bin/elements.bin
	cosy $<;
bin/support/%.bin: src/support/%.fox setups.bld
	cosy $<;


BNL/% : scripts/BNL/%.fox support.bld
	cosy $<;
EB/% : scripts/EB/%.fox support.bld
	cosy $<;
TE/% : test/%.fox support.bld
	cosy $<;

clean:
	$(call RM, '.', '*.lis')

uninstall:
	$(call RM, 'bin', '*.bin')
	rm -f *.bld
	$(call RM, '.', '*.lis')

purge:
	$(call RM, 'data', '*.dat')
	$(call RM, 'data', '*.in')
	$(call RM, 'img', '*.png')
	$(call RM, 'img', '*.pdf')

.PHONY: clean uninstall purge
