<script setup>
import { ref } from "vue";
import {Button} from "@/components/ui/button/index.js";

const props = defineProps({
  deck: { type: Object, required: true },
  coordinates: { type: Object, required: true },
})

const open = ref(false);

const showMenu = () => {
  open.value = true;
  props.deck.setProps({
    controller: {
      dragPan: false,
      dragRotate: false,
      scrollZoom: false,
      doubleClickZoom: false,
      touchZoom: false,
      touchRotate: false
    }
  });
}

const closeMenu = () => {
  open.value = false;
  props.deck.setProps({
    controller: {
      dragPan: true,
      dragRotate: true,
      scrollZoom: true,
      doubleClickZoom: true,
      touchZoom: true,
      touchRotate: true,
    }
  })
}

defineExpose({
  showMenu,
  closeMenu,
})

</script>

<template>
  <div v-if="open" class="fixed size-5 bg-red-500 z-99 rounded-2xl" :style="`left: ${coordinates.x}px; top: ${coordinates.y}px`"></div>

  <div v-if="open" class="fixed z-99 p-5 w-50 rounded-lg bg-white" :style="`left: ${coordinates.x + 25}px; top: ${coordinates.y + 10}px`">
    <ul class="flex flex-col">
      <li>TEST: 30</li>
      <li>TEST: 30s</li>
      <li>TEST: 20</li>
      <li>TEST: 10</li>
    </ul>
    <Button @click="closeMenu">
      CLOSE
    </Button>
  </div>
</template>
