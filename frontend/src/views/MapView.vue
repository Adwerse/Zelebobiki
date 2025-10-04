<script setup>
import { ref } from 'vue';
import { Deck, TileLayer, BitmapLayer, MapView } from 'deck.gl';
import PopUp from '@/components/map/PopUp.vue'

const deckContainer = ref(null);
const coords = ref(null);
const deck = ref(null);
const popUp = ref(null);

const INITIAL_VIEW_STATE = {
  longitude: 0,
  latitude: 0,
  zoom: 2,
  pitch: 0,
  bearing: 0
};

const oceanBasemap = new TileLayer({
  id: 'ocean-basemap',
  data: 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
  minZoom: 0,
  maxZoom: 13,
  tileSize: 256,

  renderSubLayers: props => {
    const {
      bbox: {west, south, east, north}
    } = props.tile;
    return new BitmapLayer(props, {
      data: null,
      image: props.data,
      bounds: [west, south, east, north]
    });
  }
});

deck.value = new Deck({
  parent: deckContainer.value,
  initialViewState: INITIAL_VIEW_STATE,
  controller: true,
  views: [new MapView({repeat: true})],
  layers: [oceanBasemap],
  onClick: info => {
    popUp.value.showMenu();
    coords.value = {
      x: info.x,
      y: info.y,
    };
  },
});
</script>

<template>
  <div>
    <div ref="deckContainer"></div>
    <PopUp ref="popUp" :deck="deck" :coordinates="coords"/>
  </div>
</template>
