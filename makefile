
CORE = cosy.bin utilities.bin elements.bin
SETUPS = $(addsuffix .bin, BNL EB BNL8gsd)
SUPPORT = BNL/decoh_optim.bin EB/decoh_optim.bin

define RM
	find $(1) -type f -name $(2) -print -delete
endef

core.bld: $(addprefix bin/core/, $(CORE))
	echo $(CORE) >> core.bld

setups.bld: $(addprefix bin/setups/, $(SETUPS)) core.bld
	echo $(SETUPS) >> setups.bld

support.bld: $(addprefix bin/support/, $(SUPPORT)) setups.bld
	echo $(SUPPORT) >> support.bld

bin/core/cosy.bin: src/core/cosy.fox ;
	cosy $<;
bin/core/utilities.bin: src/core/utilities.fox bin/core/cosy.bin 
	cosy $<;
bin/core/elements.bin: src/core/elements.fox bin/core/utilities.bin 
	cosy $<;

bin/setups/%.bin: src/setups/%.fox bin/core/elements.bin 
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
